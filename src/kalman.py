"""Entrada compatible con el pipeline del equipo."""
import numpy as np
from .filtro_kalman import ejecutar_experimento, main


def correr_escenario_D(seed=0, paso_medicion=1):
    """Devuelve tiempo, verdad, estimación y desviación como antes."""
    if paso_medicion != 1:
        raise ValueError('Este ejemplo sencillo usa una medición en cada paso.')
    r = ejecutar_experimento(semilla=seed)
    sd = np.sqrt(np.maximum(np.diagonal(r['P'], axis1=1, axis2=2), 0))
    return r['t'], r['verdad'], r['estimados'], sd


if __name__ == '__main__':
    main()
