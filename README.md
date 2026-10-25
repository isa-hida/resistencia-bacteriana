# Resistencia bacteriana: Euler + Red bayesiana + Filtro de Kalman

Proyecto del caso de estudio "De la dosis a la resistencia".

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución rápida
```bash
python -m src.euler_model      # escenario B por defecto, guarda figures/euler_B.png
python -m src.kalman           # escenario D nominal: figuras en figures/persona3/
python -m src.bayes_net        # consulta de ejemplo sobre el DAG
python -m src.pipeline         # integración completa
```

## Estructura
| Carpeta | Contenido | Responsable principal |
|---|---|---|
| `src/euler_model.py` | Ecuaciones en diferencias (S, T, R) | Coordinador/a de modelo |
| `src/bayes_net.py` | DAG y tablas condicionales | Analista bayesiano/a |
| `src/filtro_kalman.py` | Filtro de Kalman (EKF) | Ingeniero/a de estimación |
| `src/kalman.py` | Entrada compatible con el pipeline | Ingeniero/a de estimación |
| `src/pipeline.py` | Bayes → Euler → Kalman → decisión | Todos (fase 5) |
| `params.yaml` | Parámetros nominales y escenarios A–E | Analista de decisiones |
| `bitacoras/` | Una bitácora por integrante | Cada quien |
| `docs/` | RACI, reglas, actas | Todos |

## Reglas de colaboración (resumen)
1. Una rama por persona: `feature/<nombre>-<tema>`. Nada va a `main` sin Pull Request.
2. Todo PR lo revisa y **ejecuta** otra persona antes de aprobarse.
3. Los parámetros se cambian SOLO en `params.yaml` (nunca hardcodeados).
4. Commits con tu propio usuario de Git (`git config user.name` / `user.email`).
5. Cada semana, alguien distinto expone un componente que no lidera (ver `docs/actas.md`).

## Notas importantes sobre el modelo
- Con escenarios de ciclos de 3 h se necesita `dt <= 1/24` día; el `dt = 0.1` d del PDF
  (2.4 h) no resuelve la ventana de exposición. Aquí se usa `dt_horas` por escenario.
- Las poblaciones abarcan ~7 órdenes de magnitud: en el filtro de Kalman, cuidado con la
  escala de Q y R (ver preguntas abiertas 3 y 4 del documento).
- Las referencias `[citation:N]` del PDF no tienen bibliografía: pídanla al docente.

## Figuras y datos vigentes

- `figures/euler_A.png`, `euler_B.png`, `euler_C.png`: figuras de la parte de Euler.
- `figures/persona3/sin_medir_T.png`: Kalman **nominal**, midiendo S y R.
- `figures/persona3/ajustado_sin_T.png`: ajuste experimental de Q y R, generado
  al ejecutar [el notebook de Kalman](notebooks/04_kalman_estimacion.ipynb).
- `data/persona3/sin_medir_T.csv`: datos **nominales**.
- `data/persona3/ajustado_sin_T.csv`: datos del ajuste experimental sin medir T.

