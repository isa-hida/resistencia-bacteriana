"""Modelo de subpoblaciones S, T, R en el formato de EdoLib: f(t, x, *args)."""
import numpy as np


def conc_constante(C):
    return lambda t: C


def conc_ciclica(C0=5.0, horas_exp=3.0):
    def conc(t):
        # +1e-9 dias compensa la deriva de punto flotante de t0 += dt
        # (sin esto se pierde un paso de exposicion en algunos dias)
        hora = ((t + 1e-9) % 1.0) * 24.0
        return C0 if hora < horas_exp else 0.0
    return conc


def get_subpoblaciones(t, x, p, conc):
    """Ecuaciones (1)-(3) del documento. x = [S, T, R]; p = dict de parametros; conc = funcion C(t)."""
    S, T, R = x
    C = conc(t)
    g = 1.0 - (S + T + R) / p["K"]
    dS = p["rS"] * S * g - p["dS"] * C * S - p["muST"] * S
    dT = p["rT"] * T * g - p["dT"] * C * T + p["muST"] * S - p["muTR"] * T
    dR = p["rR"] * R * g - p["dR"] * C * R + p["muTR"] * T
    return np.array([dS, dT, dR])


def metricas(t, X):
    N = X.sum(axis=1)
    fracR = X[:, 2] / N
    dom = np.argmax(fracR > 0.5) if (fracR > 0.5).any() else None
    ext = np.argmax(X[:, 0] < 1.0) if (X[:, 0] < 1.0).any() else None
    return {"fraccion_R_final": float(fracR[-1]),
            "t_dominancia_R_dias": None if dom is None else float(t[dom]),
            "t_extincion_S_dias": None if ext is None else float(t[ext])}