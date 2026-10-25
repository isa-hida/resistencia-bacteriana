
import numpy
import matplotlib.pyplot as pl
numpy.seterr(divide='ignore', invalid='ignore')


def campo_vectorial(n, x0, xf, ejex, y0, yf, ejey, ecuacion, *args):
    x = numpy.linspace(x0, xf, n)
    y = numpy.linspace(y0, yf, n)
    xx, yy = numpy.meshgrid(x, y)
    uu, vv = ecuacion(0, (xx, yy), *args)
    pl.quiver(xx, yy, uu, vv, color="blue")
    pl.xlabel(ejex)
    pl.ylabel(ejey)


def campo_direcciones(n, x0, xf, ejex, y0, yf, ejey, ecuacion, *args):
    x = numpy.linspace(x0, xf, n)
    y = numpy.linspace(y0, yf, n)
    xx, yy = numpy.meshgrid(x, y)
    uu, vv = ecuacion(0, (xx, yy), *args)
    norm = numpy.sqrt(uu ** 2 + vv ** 2)
    uu = uu / norm
    vv = vv / norm
    pl.quiver(xx, yy, uu, vv, color="blue")
    pl.xlabel(ejex)
    pl.ylabel(ejey)


def campo_pendientes(n, t0, tf,  y0, yf, ecuacion, *args):
    t = numpy.linspace(t0, tf, n)
    y = numpy.linspace(y0, yf, n)
    Y, T = numpy.meshgrid(y, t)
    u = 1
    v = ecuacion(T, Y, *args)
    normal = numpy.sqrt(u ** 2 + v ** 2)
    U = u / normal
    V = v / normal
    pl.quiver(T, Y, U, V, normal)
    pl.grid()
    # pl.xlabel(ejex)
    # pl.ylabel(ejey)


def euler(t0, tf, presente, direccion, intervalo, *args):
    futuros = []
    tiempos = []
    while True:
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        pendiente = direccion(t0, presente, *args)
        presente += pendiente * intervalo
        futuros.append(presente)
        tiempos.append(t0)
        t0 += intervalo
        if t0 >= tf:
            break
    futuros = numpy.array(futuros)
    tiempos = numpy.array(tiempos)
    return futuros, tiempos


def heun(t0, tf, presente, direccion, intervalo, *args):
    historial = []
    tiempos = []
    while True:
        historial.append(presente)
        tiempos.append(t0)
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        k1 = direccion(t0, presente, *args)
        k2 = direccion((t0 + intervalo), presente + (intervalo * k1), *args)
        pendiente = ((1.0 / 2.0) * (k1 + k2))
        presente = presente + pendiente * intervalo
        t0 = t0 + intervalo
        if t0 >= tf:
            break
    return numpy.array(historial), numpy.array(tiempos)


def Rk2(t0, tf, presente, direccion, intervalo, *args):
    historial = []
    tiempos = []
    while True:
        historial.append(presente)
        tiempos.append(t0)
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        k1 = direccion(t0, presente, *args)
        k2 = direccion((t0 + (intervalo / 2.0)), (presente + intervalo / 2.0 * k1), *args)
        presente = presente + k2 * intervalo
        t0 = t0 + intervalo
        if t0 >= tf:
            break
    return numpy.array(historial), numpy.array(tiempos)


def Rk4(t0, tf, presente, direccion, intervalo, *args):
    historial = []
    tiempos = []
    while True:
        historial.append(presente)
        tiempos.append(t0)
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        k1 = direccion(t0, presente, *args)
        k2 = direccion((t0 + (intervalo / 2.0)), (presente + ((intervalo / 2.0) * k1)), *args)
        k3 = direccion((t0 + (intervalo / 2.0)), (presente + ((intervalo / 2.0) * k2)), *args)
        k4 = direccion((t0 + intervalo), (presente + (intervalo * k3)), *args)
        pendiente = (1.0 / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4)
        presente = presente + pendiente * intervalo
        t0 = t0 + intervalo
        if t0 >= tf:
            break
    return numpy.array(historial), numpy.array(tiempos)


# ======================================================================
# AGREGADOS (Coordinador/a de modelo) -- para estados VECTORIALES [S, T, R]
# ======================================================================

def euler_vec(t0, tf, presente, direccion, intervalo, *args):
    """Euler explicito para estados vectoriales.
    Misma convencion que heun y Rk4: guarda el estado ANTES de cada paso."""
    presente = numpy.array(presente, dtype=float)   # copia: no muta la condicion inicial
    historial = []
    tiempos = []
    while True:
        historial.append(presente)
        tiempos.append(t0)
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        pendiente = direccion(t0, presente, *args)
        presente = presente + pendiente * intervalo  # NO usar += (mutaria el array ya guardado)
        t0 = t0 + intervalo
        if t0 >= tf:
            break
    return numpy.array(historial), numpy.array(tiempos)


def euler_positivo(t0, tf, presente, direccion, intervalo, *args):
    """Igual que euler_vec, pero impone x >= 0 en cada paso."""
    presente = numpy.array(presente, dtype=float)
    historial = []
    tiempos = []
    while True:
        historial.append(presente)
        tiempos.append(t0)
        if (t0 + intervalo) > tf:
            intervalo = tf - t0
        pendiente = direccion(t0, presente, *args)
        presente = numpy.maximum(0.0, presente + pendiente * intervalo)
        t0 = t0 + intervalo
        if t0 >= tf:
            break
    return numpy.array(historial), numpy.array(tiempos)


def rk4_paso(t, x, direccion, h, *args):
    k1 = direccion(t, x, *args)
    k2 = direccion(t + h / 2.0, x + h / 2.0 * k1, *args)
    k3 = direccion(t + h / 2.0, x + h / 2.0 * k2, *args)
    k4 = direccion(t + h, x + h * k3, *args)
    return x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def referencia_por_tramos(dias, x0, direccion, dt, p, C0=5.0, horas_exp=3.0):
    """Solucion de REFERENCIA del esquema ciclico: Rk4 por tramos con C constante
    (exposicion / recuperacion), para no evaluar la discontinuidad de C(t) dentro de un paso.
    Devuelve el estado al final de cada tramo."""
    he = horas_exp / 24.0
    x = numpy.array(x0, dtype=float)
    estados = [x]
    tiempos = [0.0]
    for d in range(int(dias)):
        for ini, fin, C in [(d, d + he, C0), (d + he, d + 1.0, 0.0)]:
            conc = lambda t, C=C: C
            n = max(1, int(numpy.ceil((fin - ini) / dt)))
            h = (fin - ini) / n
            t = ini
            for _ in range(n):
                x = rk4_paso(t, x, direccion, h, p, conc)
                t += h
            estados.append(x)
            tiempos.append(fin)
    return numpy.array(estados), numpy.array(tiempos)