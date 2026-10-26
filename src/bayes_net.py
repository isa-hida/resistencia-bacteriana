from pathlib import Path
from itertools import product
import csv
import numpy as np
import yaml
from .euler_model import simulate

ROOT = Path(__file__).resolve().parents[1]
FACTORES = ("Concentracion", "Duracion", "Frecuencia", "TolerPrevia",
            "TasaMutacion", "CosteFitness")


def configuracion():
    with open(ROOT / "params.yaml", encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def tasas(evidencia, cfg=None):
    """Selecciona tasas ya justificadas por el perfil; no convierte riesgo en tasa."""
    cfg = cfg or configuracion()
    perfil = evidencia["TasaMutacion"]
    if perfil not in (0, 1):
        raise ValueError("El perfil de mutacion debe ser 0 o 1.")
    origen = cfg["nominal"] if perfil == 0 else cfg["scenarios"]["C"]["override"]
    return {nombre: origen[nombre] for nombre in ("muST", "muTR")}


def experimento(p, sc, cfg=None):
    """Probabilidad de al menos una transicion nueva durante el regimen."""
    cfg = cfg or configuracion()
    t, X, C, dt = simulate(p=p, sc=sc)
    volumen = cfg["bayes"]["volumen_ml"]
    if volumen <= 0:
        raise ValueError("El volumen del evento debe ser positivo.")
    probabilidades = {}
    for evento, tasa, columna in [("ProbTolerancia", "muST", 0),
                                  ("ProbResistencia", "muTR", 1)]:
        esperados = volumen * p[tasa] * np.sum(X[:-1, columna] * np.diff(t))
        probabilidades[evento] = float(1 - np.exp(-esperados))
    return t, X, C, probabilidades


def construir(cfg=None):
    """Una fila de CPT por combinacion: 2**6 = 64 simulaciones."""
    cfg = cfg or configuracion()
    b = cfg["bayes"]
    tabla = []
    for estados in product([0, 1], repeat=6):
        evidencia = dict(zip(FACTORES, estados))
        valores = {f: b["niveles"][f][evidencia[f]] for f in FACTORES}
        p = dict(cfg["nominal"])
        p.update(tasas(evidencia, cfg))
        total = p["x0"]["S"] + p["x0"]["T"]
        p["x0"] = {"S": total * (1-valores["TolerPrevia"]),
                   "T": total * valores["TolerPrevia"], "R": p["x0"]["R"]}
        p["rR"] = p["rS"] * (1-valores["CosteFitness"])
        sc = dict(esquema="ciclico", C=valores["Concentracion"],
                  duracion_dias=valores["Duracion"], dt_horas=b["dt_horas"],
                  horas_exposicion=b["horas_exposicion"],
                  periodo_horas=24/valores["Frecuencia"])
        _, _, _, probabilidades = experimento(p, sc, cfg)
        tabla.append(dict(evidencia, **probabilidades))
    return tabla


def consulta(evidencia=None, tabla=None, variable="ProbResistencia", cfg=None):
    """Promedia las filas compatibles; las variables desconocidas se omiten."""
    cfg = cfg or configuracion()
    evidencia = {} if evidencia is None else evidencia
    if any(f not in FACTORES or v not in (0, 1) for f, v in evidencia.items()):
        raise ValueError("Usa factores conocidos y estados 0 o 1.")
    if variable not in ("ProbResistencia", "ProbTolerancia"):
        raise ValueError("Consulta resistencia o tolerancia.")
    tabla = construir(cfg) if tabla is None else tabla
    prior = cfg["bayes"]["prior_estado_1"]
    if not 0 < prior < 1:
        raise ValueError("El prior debe estar entre cero y uno.")
    numerador = denominador = 0.0
    for fila in tabla:
        if all(fila[f] == v for f, v in evidencia.items()):
            peso = np.prod([prior if fila[f] else 1-prior for f in FACTORES])
            numerador += peso * fila[variable]
            denominador += peso
    if denominador == 0:
        raise ValueError("No hay filas compatibles con la evidencia.")
    return float(numerador / denominador)


if __name__ == "__main__":
    tabla = construir()
    carpeta = ROOT / "data" / "bayes"
    carpeta.mkdir(parents=True, exist_ok=True)
    with open(carpeta / "cpt_simuladas.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(tabla[0]))
        escritor.writeheader()
        escritor.writerows(tabla)
    evidencia = {"Concentracion": 1, "Duracion": 0, "Frecuencia": 1, "TolerPrevia": 0}
    print("P(adquisicion nueva de resistencia | 3 h diarias, 10 dias, C alta, T baja):")
    print(consulta(evidencia, tabla))
