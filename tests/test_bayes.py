"""Pruebas pequenas para la version sencilla: python -m unittest discover -s tests -v."""
import unittest
from unittest.mock import patch
import numpy as np
from src.bayes_net import configuracion, construir, consulta, tasas, experimento, FACTORES
from src.pipeline import ejecutar_pipeline
from src.euler_model import load_params
from src.filtro_kalman import filtrar


class BayesSencillo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tabla = construir()

    def test_64_combinaciones_y_probabilidades_validas(self):
        self.assertEqual(len(self.tabla), 64)
        self.assertEqual(len({tuple(fila[f] for f in FACTORES) for fila in self.tabla}), 64)
        for fila in self.tabla:
            self.assertTrue(0 <= fila['ProbResistencia'] <= 1)
            self.assertTrue(0 <= fila['ProbTolerancia'] <= 1)

    def test_probabilidad_total_calculada_a_mano(self):
        tabla = [dict(fila, ProbResistencia=.8 if fila['Concentracion'] else .1)
                 for fila in self.tabla]
        self.assertAlmostEqual(consulta({}, tabla), .5*.8 + .5*.1)
        self.assertAlmostEqual(consulta({'Concentracion': 1}, tabla), .8)

    def test_consulta_del_pdf_promedia_cuatro_filas(self):
        e = dict(Concentracion=1, Duracion=0, Frecuencia=1, TolerPrevia=0)
        compatibles = [f['ProbResistencia'] for f in self.tabla
                       if all(f[k] == v for k, v in e.items())]
        self.assertEqual(len(compatibles), 4)
        self.assertAlmostEqual(consulta(e, self.tabla), sum(compatibles)/4)

    def test_evidencia_invalida(self):
        for e in [{'Duracion': 3}, {'OtraVariable': 1}]:
            with self.assertRaises(ValueError): consulta(e, self.tabla)

    def test_resistentes_iniciales_no_son_nuevas_transiciones(self):
        p, sc = load_params('A')
        p['muTR'] = 0.
        _, X, _, prob = experimento(p, sc)
        self.assertGreater(X[-1, 2], 0.)
        self.assertEqual(prob['ProbResistencia'], 0.)

    def test_tasas_corresponden_al_perfil_del_equipo(self):
        cfg = configuracion()
        for estado, origen in [(0, cfg['nominal']), (1, cfg['scenarios']['C']['override'])]:
            self.assertEqual(tasas({'TasaMutacion': estado}),
                             {k: origen[k] for k in ('muST', 'muTR')})

    def test_pipeline_comparte_tasas_y_concentracion_con_kalman(self):
        with patch('src.pipeline.filtrar', wraps=filtrar) as filtro:
            r = ejecutar_pipeline('B', horizonte=1.)
        args = filtro.call_args.args
        self.assertEqual(args[4]['muTR'], r['resumen']['tasas']['muTR'])
        np.testing.assert_array_equal(args[5], r['C'])
        self.assertEqual(r['X'].shape, r['estimados'].shape)

    def test_E_no_inventa_datos(self):
        with self.assertRaises(ValueError): ejecutar_pipeline('E')


if __name__ == '__main__':
    unittest.main()
