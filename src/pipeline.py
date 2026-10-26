"""Conexion sencilla: tasas del perfil -> Euler -> Kalman, con el mismo escenario."""
import argparse
import csv
import json
import numpy as np
from .bayes_net import configuracion, tasas, experimento
from .euler_model import load_params, metricas, ROOT
from .filtro_kalman import generar_mediciones, filtrar, rmse


def ejecutar_pipeline(escenario="B", horizonte=None):
    cfg = configuracion()
    b = cfg["bayes"]
    if escenario not in b["perfil_escenario"]:
        raise ValueError("E necesita datos reales y un regimen numerico definido.")
    p, sc = load_params(escenario)
    if horizonte is not None:
        if horizonte <= 0:
            raise ValueError("La duracion debe ser positiva.")
        sc["duracion_dias"] = horizonte
    perfil = b["perfil_escenario"][escenario]
    p.update(tasas({"TasaMutacion": perfil}, cfg))
    t, X, C, probabilidades = experimento(p, sc, cfg)
    H = np.array([[1., 0., 0.], [0., 0., 1.]])
    kc = cfg["scenarios"]["D"]["kalman"]
    medidas, R = generar_mediciones(X, H, kc["sigma_obs_rel"], b["semilla"])
    x0 = X[0] * np.array(b["factor_inicial_kalman"])
    P0 = np.diag((b["sd_inicial_kalman"] * x0)**2)
    estimados, P = filtrar(t, medidas, R, H, p, C, x0, P0, kc["q_rel"])
    resumen = dict(escenario=escenario, dias=sc["duracion_dias"],
                   origen="simulacion didactica, no calibracion experimental",
                   tasas={k: p[k] for k in ("muST", "muTR")},
                   probabilidades=probabilidades, metricas=metricas(t, X),
                   fraccion_R_EKF=float(estimados[-1, 2]/estimados[-1].sum()),
                   RMSE=rmse(X, estimados).tolist())
    return dict(resumen=resumen, t=t, X=X, C=C, estimados=estimados, P=P,
                parametros=p, medidas=medidas, R_medicion=R, H=H)


def guardar(resultados):
    carpeta = ROOT / "results" / "bayes"
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = "pipeline_" + resultados["resumen"]["escenario"]
    (carpeta / (nombre + ".json")).write_text(
        json.dumps(resultados["resumen"], indent=2), encoding="utf-8")
    columnas = np.column_stack([resultados["t"], resultados["X"], resultados["estimados"]])
    np.savetxt(carpeta / (nombre + ".csv"), columnas, delimiter=",", comments="",
               header="dias,S_Euler,T_Euler,R_Euler,S_EKF,T_EKF,R_EKF")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--escenario", choices=list("ABCDE"), default="B")
    parser.add_argument("--horizonte", type=float)
    args = parser.parse_args()
    resultados = ejecutar_pipeline(args.escenario, args.horizonte)
    guardar(resultados)
    print(json.dumps(resultados["resumen"], indent=2))
