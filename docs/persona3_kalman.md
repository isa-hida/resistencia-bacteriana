# Persona 3: filtro de Kalman

`python -m src.kalman` genera el caso **nominal** sin medir T:
`figures/persona3/sin_medir_T.png` y `data/persona3/sin_medir_T.csv`. Usa `q_rel=0.02` del escenario D en `params.yaml` y
`factor_R=1`. El código del EKF está en `src/filtro_kalman.py`.

El trabajo explicado está en `notebooks/04_kalman_estimacion.ipynb`.
Euler proporciona la trayectoria real sintética. Se miden S y R y se estima T sin observarla. F es el jacobiano
numérico del avance de Euler; Q representa incertidumbre del modelo, R la
del sensor y P la del estado estimado.

El notebook compara 20 combinaciones de Q y R con las semillas 0–4. Elige
la de menor RMSE de T entre las que dan una cobertura empírica de 93.5% a
96.5%, e informa también la anchura media del intervalo. Las semillas 5–9
sirven para verificar error, cobertura y convergencia sin participar en
la elección. Los parámetros ajustados se guardan por separado en
`data/persona3/ajustado_sin_T.csv` y `figures/persona3/ajustado_sin_T.png`.

Este ajuste es válido solo para el simulador empleado. Como la trayectoria
real y la predicción usan el mismo Euler determinista, Q puede resultar cero.
Eso no implica ruido de proceso nulo en un experimento de laboratorio.
Tampoco identifica Q y R verdaderas: con datos reales habría que estimar
la incertidumbre de las mediciones y comprobar de nuevo los intervalos.

El notebook también pronostica S, T y R a 24 horas desde el día 10, sin usar
mediciones futuras, y compara la fracción resistente prevista con Euler.
