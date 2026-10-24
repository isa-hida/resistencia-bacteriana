"""Integración: Bayes -> Euler -> Kalman -> decisión (fase 5). Esqueleto funcional."""
from .euler_model import load_params, simulate, metricas
from .bayes_net import consulta
from .kalman import correr_escenario_D


def main():
    # 1) Bayes: probabilidad de resistencia dado el régimen
    q = consulta({"Concentracion": 1, "Duracion": 0, "Frecuencia": 1})
    p_res = float(q.values[1])
    print(f"[Bayes] P(resistencia | régimen) = {p_res:.3f}")
    # 2) Euler: TODO escalar muST/muTR según p_res y simular
    p, sc = load_params("B")
    t, X, C, dt = simulate("B", p, sc)
    print("[Euler] métricas:", metricas(t, X))
    # 3) Kalman: estimar composición con mediciones ruidosas
    t, Xt, est, sd = correr_escenario_D()
    print("[Kalman] fracción R estimada final:",
          float(est[-1, 2] / est[-1].sum()))
    # 4) Decisión: TODO comparar regímenes y elegir el que minimice resistencia


if __name__ == "__main__":
    main()
