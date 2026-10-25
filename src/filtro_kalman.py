"""Persona 3: estimar S, T y R con un filtro de Kalman extendido (EKF)."""
import numpy as np
from .euler_model import load_params, simulate, step, concentracion
from pathlib import Path
import matplotlib.pyplot as plt


def generar_mediciones(verdad, H, ruido=0.10, semilla=0):
    """Simula y = H x + ruido gaussiano; R guarda la varianza de cada sensor."""
    esperado = verdad @ H.T
    desviacion = ruido * np.maximum(np.abs(esperado), 1.0)
    generador = np.random.default_rng(semilla)
    medidas = esperado + generador.normal(size=esperado.shape) * desviacion
    R = np.array([np.diag(fila**2) for fila in desviacion])
    return medidas, R


def jacobiano(funcion, estado):
    """Aproxima cómo cambia el próximo estado al variar S, T o R."""
    F = np.zeros((3, 3))
    for j in range(3):
        h = max(abs(estado[j]) * 1e-5, 1e-4)
        arriba, abajo = estado.copy(), estado.copy()
        arriba[j] += h
        abajo[j] -= h
        F[:, j] = (funcion(arriba) - funcion(abajo)) / (2*h)
    return F


def filtrar(t, medidas, R, H, parametros, concentraciones, x0, P0, q_rel=0.02):
    """EKF: predecir con Euler y corregir con las mediciones.

    No recibe la trayectoria verdadera. C[k] es la dosis aplicada al paso k-1 -> k
    en el Euler ACTUAL. Se usa el mismo paso de tiempo que devuelve Euler.
    """
    x, P = x0.copy(), P0.copy()
    estimados = np.zeros((len(t), 3))
    covarianzas = np.zeros((len(t), 3, 3))
    I = np.eye(3)
    for k in range(len(t)):
        if k > 0:
            dt = t[k] - t[k-1]
            avance = lambda estado: step(estado, concentraciones[k], dt, parametros)
            F = jacobiano(avance, x)
            x = avance(x)                                      # predicción del estado
            Q = np.diag((q_rel * np.maximum(x, 1.0))**2)
            P = F @ P @ F.T + Q                                # predicción de incertidumbre

        diferencia = medidas[k] - H @ x                       # medición - predicción
        S = H @ P @ H.T + R[k]                                 # incertidumbre de la diferencia
        K = np.linalg.solve(S, H @ P).T                       # ganancia de Kalman
        x = np.maximum(x + K @ diferencia, 0.0)               # corrección del estado
        A = I - K @ H
        P = A @ P @ A.T + K @ R[k] @ K.T                      # corrección de incertidumbre
        estimados[k], covarianzas[k] = x, P
    return estimados, covarianzas


def predecir(estado, t_inicio, pasos, dt, parametros, escenario):
    """Pronostica pasos futuros con Euler, sin usar mediciones futuras."""
    x = estado.copy()
    for i in range(pasos):
        C = concentracion(t_inicio + i * dt, escenario)
        x = step(x, C, dt, parametros)
    return x


def rmse(verdad, valores):
    """Error cuadrático medio por población, en UFC/mL."""
    return np.sqrt(np.mean((verdad - valores)**2, axis=0))


def ejecutar_experimento(medir_T=False, ruido=None, semilla=0, q_rel=None, factor_R=1.0):
    """Conecta el EKF con el escenario D de Euler; H=I o mide solo S y R."""
    parametros, escenario = load_params('D')
    if ruido is None:
        ruido = escenario['kalman']['sigma_obs_rel']
    if q_rel is None:
        q_rel = escenario['kalman']['q_rel']
    t, verdad, C, _ = simulate('D', parametros, escenario)
    H = np.eye(3) if medir_T else np.array([[1., 0., 0.], [0., 0., 1.]])
    medidas, R = generar_mediciones(verdad, H, ruido, semilla)
    if factor_R <= 0:
        raise ValueError('factor_R debe ser positivo')
    R_filtro = factor_R * R  # varianza supuesta por el filtro; R es la real sintética
    x0 = verdad[0] * np.array([1.2, 0.8, 1.3])  # estimación inicial imperfecta
    P0 = np.diag((0.5 * np.maximum(x0, 1.0))**2)
    estimados, P = filtrar(t, medidas, R_filtro, H, parametros, C, x0, P0, q_rel)
    return {'t': t, 'verdad': verdad, 'medidas': medidas, 'estimados': estimados,
            'P': P, 'H': H, 'R': R, 'R_filtro': R_filtro}


def graficar(resultado):
    """Una gráfica por población: verdad, medición, EKF e incertidumbre."""
    t, verdad = resultado['t'], resultado['verdad']
    medidas, estimados = resultado['medidas'], resultado['estimados']
    P = resultado['P']
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
    for j, eje in enumerate(ejes):
        eje.plot(t, verdad[:, j], 'k', label='Real (Euler)')
        sensor = np.where(resultado['H'][:, j] == 1)[0]
        if len(sensor):
            eje.scatter(t, medidas[:, sensor[0]], s=5, alpha=.2, label='Medido')
        eje.plot(t, estimados[:, j], '--', label='EKF')
        sd = np.sqrt(np.maximum(P[:, j, j], 0))
        eje.fill_between(t, estimados[:, j]-1.96*sd, estimados[:, j]+1.96*sd,
                         alpha=.12, label='Intervalo aprox. 95%')
        eje.set(title=('S', 'T', 'R')[j], xlabel='Días', ylabel='UFC/mL')
        eje.set_yscale('symlog', linthresh=1)
        eje.grid(alpha=.2)
        eje.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    carpeta = Path(__file__).resolve().parents[1] / 'figures/persona3'
    carpeta.mkdir(parents=True, exist_ok=True)
    datos = carpeta.parents[1] / 'data/persona3'
    datos.mkdir(parents=True, exist_ok=True)
    r = ejecutar_experimento(medir_T=False)
    print('RMSE EKF [S,T,R]:', rmse(r['verdad'], r['estimados']))
    medidas = np.full_like(r['verdad'], np.nan)
    medidas[:, 0], medidas[:, 2] = r['medidas'][:, 0], r['medidas'][:, 1]
    tabla = np.column_stack([r['t'], r['verdad'], medidas, r['estimados']])
    np.savetxt(datos / 'sin_medir_T.csv', tabla, delimiter=',', comments='',
               header='dias,S_real,T_real,R_real,S_medida,T_medida,R_medida,S_EKF,T_EKF,R_EKF')
    figura = graficar(r)
    figura.savefig(carpeta / 'sin_medir_T.png', dpi=150)
    plt.close(figura)


if __name__ == '__main__':
    main()
