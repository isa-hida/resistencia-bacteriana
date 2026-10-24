"""Capa 2: red bayesiana de condiciones de tratamiento -> probabilidad de resistencia.

!! Las CPDs de abajo son PLACEHOLDERS ilustrativas. El/la analista bayesiano/a debe
   reemplazarlas con valores justificados con datos o literatura. !!
Estados discretos: 0 = bajo, 1 = alto.
"""
import itertools
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

EDGES = [
    ("Concentracion", "ExtincionSensibles"),
    ("Duracion", "ExtincionSensibles"),
    ("Frecuencia", "ExtincionSensibles"),
    ("ExtincionSensibles", "ProbResistencia"),
    ("TolerPrevia", "ProbResistencia"),
    ("TasaMutacion", "ProbResistencia"),
    ("CosteFitness", "ProbResistencia"),
]


def root(name, p_alto=0.5):
    return TabularCPD(name, 2, [[1 - p_alto], [p_alto]])


def construir():
    m = DiscreteBayesianNetwork(EDGES)
    cpds = [root(n) for n in ["Concentracion", "Duracion", "Frecuencia",
                              "TolerPrevia", "TasaMutacion", "CosteFitness"]]
    cols = list(itertools.product([0, 1], repeat=3))
    p_ext = [0.1 + 0.25 * sum(c) for c in cols]                    # PLACEHOLDER
    cpds.append(TabularCPD("ExtincionSensibles", 2,
                [[1 - v for v in p_ext], p_ext],
                evidence=["Concentracion", "Duracion", "Frecuencia"],
                evidence_card=[2, 2, 2]))
    cols = list(itertools.product([0, 1], repeat=4))
    p_res = [min(0.95, 0.05 + 0.2 * c[0] + 0.2 * c[1] + 0.2 * c[2] - 0.15 * c[3])
             for c in cols]                                         # PLACEHOLDER
    p_res = [max(0.01, v) for v in p_res]
    cpds.append(TabularCPD("ProbResistencia", 2,
                [[1 - v for v in p_res], p_res],
                evidence=["ExtincionSensibles", "TolerPrevia", "TasaMutacion", "CosteFitness"],
                evidence_card=[2, 2, 2, 2]))
    m.add_cpds(*cpds)
    assert m.check_model()
    return m


def consulta(evidencia):
    inf = VariableElimination(construir())
    return inf.query(["ProbResistencia"], evidence=evidencia, show_progress=False)


if __name__ == "__main__":
    q = consulta({"Concentracion": 1, "Duracion": 1, "TolerPrevia": 0})
    print("P(Resistencia | conc alta, duración alta, tolerancia previa baja):")
    print(q)
