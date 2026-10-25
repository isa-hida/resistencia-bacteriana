"""Barridos de sensibilidad parametrica (Coordinador/a de modelo).

Metrica principal: tiempo hasta dominancia SOSTENIDA de resistentes
(dias desde los cuales la fraccion de R se mantiene > umbral hasta el final del horizonte).
"""
import numpy as np
from .EdoLib import euler_vec
from .modelo import get_subpoblaciones, conc_ciclica


def correr(p, dias=100, dt_horas=0.0625, C0=5.0, horas_exp=3.0):
    x0 = np.array([p["x0"]["S"], p["x0"]["T"], p["x0"]["R"]])
    X, t = euler_vec(0, dias, x0, get_subpoblaciones, dt_horas / 24.0, p,
                     conc_ciclica(C0, horas_exp))
    return t, X


def t_dominancia(t, X, umbral=0.5):
    """Primer instante desde el cual R > umbral hasta el final. NaN si no ocurre."""
    f = X[:, 2] / X.sum(axis=1)
    if f[-1] <= umbral:
        return np.nan
    por_debajo = np.where(f <= umbral)[0]
    i = 0 if len(por_debajo) == 0 else por_debajo[-1] + 1
    return float(t[i])


def t_extincion_S(t, X, umbral=1.0):
    idx = np.where(X[:, 0] < umbral)[0]
    return float(t[idx[0]]) if len(idx) else np.nan


def resumen(t, X):
    f = X / X.sum(axis=1, keepdims=True)
    i20 = np.searchsorted(t, 20.0)
    return {"t_dom_R": t_dominancia(t, X),
            "t_ext_S": t_extincion_S(t, X),
            "fracR_20d": float(f[min(i20, len(t) - 1), 2]),
            "fracT_20d": float(f[min(i20, len(t) - 1), 1])}


def barrido_2d(p, muST_vals, muTR_vals, **kw):
    """Devuelve matrices [len(muTR), len(muST)] con t_dom_R, t_ext_S, fracR_20d y fracT_20d."""
    out = {k: np.full((len(muTR_vals), len(muST_vals)), np.nan)
           for k in ["t_dom_R", "t_ext_S", "fracR_20d", "fracT_20d"]}
    for i, mtr in enumerate(muTR_vals):
        for j, mst in enumerate(muST_vals):
            q = dict(p); q["muST"] = mst; q["muTR"] = mtr
            t, X = correr(q, **kw)
            for k, v in resumen(t, X).items():
                out[k][i, j] = v
    return out


def tornado(p, nombres=("rS", "rT", "rR", "dS", "dT", "dR", "muST", "muTR", "K"),
            delta=0.2, **kw):
    """Sensibilidad local: cambia cada parametro +-delta (relativo) y mide t_dom_R.
    Devuelve dict nombre -> (t_menos, t_base, t_mas, elasticidad)."""
    t, X = correr(p, **kw)
    base = t_dominancia(t, X)
    res = {}
    for n in nombres:
        vals = []
        for s in (-delta, +delta):
            q = dict(p); q[n] = p[n] * (1 + s)
            tt, XX = correr(q, **kw)
            vals.append(t_dominancia(tt, XX))
        tm, tp = vals
        # elasticidad = (cambio relativo del resultado) / (cambio relativo del parametro)
        el = ((tp - tm) / base) / (2 * delta) if np.isfinite(tm) and np.isfinite(tp) else np.nan
        res[n] = (tm, base, tp, el)
    return res