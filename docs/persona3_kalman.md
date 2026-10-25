# Persona 3: Kalman y estimación

Ejecutar desde la raíz del repositorio:

```bash
python -m src.kalman
python -m unittest discover -s tests
```

Los CSV con datos sintéticos quedan en `data/persona3/`; las gráficas en
`figures/persona3/`. Un valor `NaN` en T_medida significa que no se midió.

El desarrollo explicado está en `notebooks/02_kalman_estimacion.ipynb`.
El código del filtro está en `src/filtro_kalman.py`.

**Estado:** `[S,T,R]` en UFC/mL. Los estados reales vienen directamente del
escenario D del Euler del equipo. Se añaden mediciones gaussianas sintéticas.
Primero se miden las tres poblaciones (`H=I`); también se ensaya medir solo
S y R. `R` es la varianza del sensor, `Q` la incertidumbre de la predicción y
`P` la incertidumbre del estado estimado. Se toman matrices diagonales.
El prior es `[1.2*S0,0.8*T0,1.3*R0]`, con desviación inicial del 50%.

El EKF usa el avance de Euler como función de transición. Calcula su jacobiano
numéricamente y aplica predicción y corrección. El RMSE compara estimación y
verdad por población, en UFC/mL. Las gráficas muestran las mediciones y un
intervalo gaussiano aproximado del 95%.

**Supuestos:** el ruido del sensor es 10% de cada población (mínimo de escala
1 UFC/mL); `R` se conoce porque los datos son sintéticos. `Q` vale 2% de la
predicción por paso. Estos valores son ilustrativos y requieren calibración
con datos reales. El estado negativo se proyecta a cero; los intervalos
son aproximados cerca de ese límite.

**Pendiente del equipo:** Euler tiene un desfase temporal en la concentración.
El filtro sigue el calendario que Euler entrega hoy para comparar estados del
mismo experimento. Hay que repetir resultados cuando la persona 1 corrija Euler.
Esta parte no toma decisiones de tratamiento ni integra Bayes.
