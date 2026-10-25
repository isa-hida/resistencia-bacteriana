"""Capa 3: filtro de Kalman (linealizado en cada paso = EKF) sobre el modelo de Euler.

F_k = I + dt * J_f(x_hat, C_k)  (jacobiano numérico de euler_model.rates)
Observaciones: conteos de placas de S y R (T no se mide directamente).
"""
import numpy as np
import matplotlib.pyplot as plt
from .euler_model import load_params, simulate, rates, step, ROOT


def jacobiano(x, C, p, eps=1e-6):
    J = np.zeros((3, 3))
    f0 = rates(x, C, p)
    for j in range(3):
        h = max(eps * abs(x[j]), 1e-3)
        xp = x.copy(); xp[j] += h
        J[:, j] = (rates(xp, C, p) - f0) / h
    return J


class KalmanPoblacion:
    def __init__(self, p, dt, H, q_rel, x0, P0):
        self.p, self.dt, self.H, self.q_rel = p, dt, H, q_rel
        self.x, self.P = x0.astype(float), P0.astype(float)

    def predict(self, C):
        F = np.eye(3) + self.dt * jacobiano(self.x, C, self.p)
        self.x = step(self.x, C, self.dt, self.p)
        Q = np.diag((self.q_rel * np.maximum(self.x, 1.0)) ** 2)
        self.P = F @ self.P @ F.T + Q

    def update(self, y, R):
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = np.maximum(0.0, self.x + K @ (y - self.H @ self.x))
        self.P = (np.eye(3) - K @ self.H) @ self.P


def correr_escenario_D(seed=0, paso_medicion=2):
    rng = np.random.default_rng(seed)
    p, sc = load_params("D")
    kc = sc["kalman"]
    t, X, C, dt = simulate("D", p, sc)
    H = np.array([[1.0, 0, 0], [0, 0, 1.0]])      # mide S y R
    sig = kc["sigma_obs_rel"]
    # "Verdad" con ruido de proceso para que filtro y planta no coincidan
    X_true = X.copy()
    for k in range(1, len(t)):
        X_true[k] = np.maximum(0, step(X_true[k-1], C[k], dt, p)
                               * (1 + kc["q_rel"] * rng.standard_normal(3)))
    x0 = np.array([p["x0"]["S"], 10 * p["x0"]["T"], 10 * p["x0"]["R"]])  # prior errado
    P0 = np.diag((0.5 * x0) ** 2)
    kf = KalmanPoblacion(p, dt, H, kc["q_rel"], x0, P0)
    est = np.zeros_like(X_true); sd = np.zeros_like(X_true); est[0] = x0
    for k in range(1, len(t)):
        kf.predict(C[k])
        if k % paso_medicion == 0:
            y = H @ X_true[k] * (1 + sig * rng.standard_normal(2))
            Rm = np.diag((sig * np.maximum(y, 1.0)) ** 2)
            kf.update(y, Rm)
        est[k] = kf.x; sd[k] = np.sqrt(np.diag(kf.P))
    return t, X_true, est, sd


if __name__ == "__main__":
    t, Xt, est, sd = correr_escenario_D()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for i, lab in enumerate(["S", "T", "R"]):
        axs[i].semilogy(t, np.maximum(Xt[:, i], 0.1), label="verdad")
        axs[i].semilogy(t, np.maximum(est[:, i], 0.1), "--", label="Kalman")
        axs[i].set(title=lab, xlabel="días"); axs[i].grid(alpha=.3)
    axs[0].legend(); fig.tight_layout()
    out = ROOT / "figures" / "kalman_D.png"; fig.savefig(out, dpi=150)
    err_T = np.abs(est[:, 1] - Xt[:, 1]) / np.maximum(Xt[:, 1], 1)
    print("Error relativo medio en T (no medida):", float(err_T.mean()))
    print("Figura:", out)
