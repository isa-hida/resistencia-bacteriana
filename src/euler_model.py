"""Capa 1: ecuaciones en diferencias (Euler explícito) para S, T, R."""
from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def _num(v):
    """Convierte recursivamente strings numéricos (p.ej. '1.0e8') a float."""
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def load_params(scenario="B"):
    with open(ROOT / "params.yaml", encoding="utf-8") as archivo:
        cfg = _num(yaml.safe_load(archivo))
    p = dict(cfg["nominal"])
    sc = dict(cfg["scenarios"][scenario])
    p.update(sc.get("override", {}))
    return p, sc


def concentracion(t_dias, sc):
    if sc["esquema"] == "constante":
        return sc["C"]
    # +1e-9 dias compensa la deriva de punto flotante (si no, se pierde un paso de exposicion)
    periodo = sc.get("periodo_horas", 24.0)
    if periodo <= 0 or not 0 <= sc["horas_exposicion"] <= periodo:
        raise ValueError("Periodo positivo y exposicion entre cero y el periodo.")
    hora = ((t_dias + 1e-9) % (periodo / 24.0)) * 24.0
    return sc["C"] if hora < sc["horas_exposicion"] else 0.0


def rates(x, C, p):
    """Derivada continua f(x, C). Útil también para linealizar en Kalman."""
    S, T, R = x
    N = S + T + R
    g = 1.0 - N / p["K"]
    dS = p["rS"] * S * g - p["dS"] * C * S - p["muST"] * S
    dT = p["rT"] * T * g - p["dT"] * C * T + p["muST"] * S - p["muTR"] * T
    dR = p["rR"] * R * g - p["dR"] * C * R + p["muTR"] * T
    return np.array([dS, dT, dR])


def step(x, C, dt, p):
    return np.maximum(0.0, x + dt * rates(x, C, p))   # no negatividad


def simulate(scenario="B", p=None, sc=None):
    if p is None or sc is None:
        p, sc = load_params(scenario)
    dt = sc["dt_horas"] / 24.0
    n = int(round(sc["duracion_dias"] / dt)) + 1
    t = np.arange(n) * dt
    X = np.zeros((n, 3))
    C = np.zeros(n)
    X[0] = [p["x0"]["S"], p["x0"]["T"], p["x0"]["R"]]
    for k in range(1, n):
        C[k] = concentracion(t[k - 1], sc)   # C vigente durante el paso k-1 -> k (inicio del paso)
        X[k] = step(X[k - 1], C[k], dt, p)
    return t, X, C, dt


def metricas(t, X):
    """Métricas derivadas de la capa 1."""
    N = X.sum(axis=1)
    fracR = X[:, 2] / np.maximum(N, 1e-12)
    dom = np.argmax(fracR > 0.5) if (fracR > 0.5).any() else None
    ext = np.argmax(X[:, 0] < 1.0) if (X[:, 0] < 1.0).any() else None
    return {
        "fraccion_R_final": float(fracR[-1]),
        "t_dominancia_R_dias": None if dom is None else float(t[dom]),
        "t_extincion_S_dias": None if ext is None else float(t[ext]),
    }


def plot(t, X, title, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, lab in enumerate(["Sensibles", "Tolerantes", "Resistentes"]):
        ax.semilogy(t, np.maximum(X[:, i], 1e-1), label=lab)
    ax.set(xlabel="Tiempo (días)", ylabel="Densidad (UFC/mL)", title=title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    import sys
    esc = sys.argv[1] if len(sys.argv) > 1 else "B"
    t, X, C, dt = simulate(esc)
    out = ROOT / "figures" / f"euler_{esc}.png"
    plot(t, X, f"Escenario {esc}", out)
    print("Métricas:", metricas(t, X))
    print("Figura:", out)
