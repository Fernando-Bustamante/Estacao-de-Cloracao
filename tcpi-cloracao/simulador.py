"""
Simulador da estacao de cloracao (TCPI) - planta nao linear, analisador de cloro
e laco de simulacao com controlador amostrado.

Estados   x = [hA, hB, hC, CC, hD, CD]             (m, m, m, mg/L, m, mg/L)
Comandos  u = [vA1, vB1, pAC, pBC, vC1, pAD, pBD, vD1]   (fracao do curso, 0 a 1)
Perturbacoes: kd (1/s), CB (mg/L), hj (m) de cada tanque de mistura.

Equacoes da Secao 2 do relatorio (eqs. de nivel, concentracao e reservatorios).
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

# indices para nao errar a ordem
HA, HB, HC, CC, HD, CD = range(6)
VA1, VB1, PAC, PBC, VC1, PAD, PBD, VD1 = range(8)
NOMES_X = ["hA", "hB", "hC", "CC", "hD", "CD"]
NOMES_U = ["vA1", "vB1", "pAC", "pBC", "vC1", "pAD", "pBD", "vD1"]


@dataclass
class Parametros:
    # tanques de mistura C e D (Tabela 4)
    R0: float = 1.0
    R1: float = 2.0
    H: float = 3.0
    K1: float = 1.0e-2                 # m3/s, bomba de agua
    K2: float = 20 / 3.6e6             # m3/s, bomba dosadora (20 L/h)
    Kv: float = 6.0e-3 / (0.5 * math.sqrt(2.0))   # 6 L/s com 50% em h = 2 m
    CB: float = 10 * 70.90 / 74.44 * 1000          # mg/L de cloro livre
    kd: float = 1 / 3600               # 1/s (1 h^-1)
    # reservatorios A e B (Tabela 4)
    AA: float = math.pi * 2.0 ** 2
    KA1: float = 2.0e-2
    AB: float = math.pi * 0.5 ** 2
    KB1: float = 40 / 3.6e6
    # PROVISORIO: o relatorio nao define altura nem nivel nominal de A e B
    HA_max: float = 4.0
    HB_max: float = 4.0
    hA_nom: float = 2.0
    hB_nom: float = 2.0
    # analisador de cloro (atraso + primeira ordem)
    theta: float = 60.0
    tau_m: float = 30.0
    # ponto de operacao dos tanques de mistura
    h_nom: float = 2.0
    C_nom: float = 2.0
    q3_nom: float = 6.0e-3


# ---------------------------------------------------------------- geometria
def raio(p, h):
    return p.R0 + (p.R1 - p.R0) / p.H * h


def area(p, h):
    return math.pi * raio(p, h) ** 2


def volume(p, h):
    return math.pi * p.H / (3 * (p.R1 - p.R0)) * (raio(p, h) ** 3 - p.R0 ** 3)


# ------------------------------------------------------- ponto de operacao
def ponto_operacao(p):
    """Equilibrio nominal (Secao 2.10). Retorna x0 (6) e u0 (8)."""
    V = volume(p, p.h_nom)
    q3 = p.q3_nom
    q2 = p.C_nom * (q3 + p.kd * V) / p.CB
    q1 = q3 - q2
    u1, u2, u3 = q1 / p.K1, q2 / p.K2, q3 / (p.Kv * math.sqrt(p.h_nom))
    vA1, vB1 = 2 * q1 / p.KA1, 2 * q2 / p.KB1
    x0 = np.array([p.hA_nom, p.hB_nom, p.h_nom, p.C_nom, p.h_nom, p.C_nom])
    u0 = np.array([vA1, vB1, u1, u2, u3, u1, u2, u3])
    return x0, u0


def perturbacoes_nominais(p):
    return {"kd_C": p.kd, "kd_D": p.kd, "CB": p.CB, "hj_C": 0.0, "hj_D": 0.0}


# ------------------------------------------------------------------ planta
def derivadas(p, x, u, d, limites=True):
    """Lado direito do modelo nao linear. Com limites=True, inclui o que o
    modelo do relatorio nao tem: bomba sem succao com reservatorio vazio e
    transbordo nos tanques cheios."""
    hA, hB, hC, Cc, hD, Cd = x
    vA1, vB1, pAC, pBC, vC1, pAD, pBD, vD1 = u

    sucA = 0.0 if (limites and hA <= 0.0) else 1.0
    sucB = 0.0 if (limites and hB <= 0.0) else 1.0
    q1C, q1D = p.K1 * pAC * sucA, p.K1 * pAD * sucA
    q2C, q2D = p.K2 * pBC * sucB, p.K2 * pBD * sucB

    dhA = (p.KA1 * vA1 - q1C - q1D) / p.AA
    dhB = (p.KB1 * vB1 - q2C - q2D) / p.AB

    def mistura(h, C, q1, q2, u3, hj, kd, CB):
        q3 = p.Kv * u3 * math.sqrt(max(h - hj, 0.0))
        dh = (q1 + q2 - q3) / area(p, max(h, 0.0))
        V = max(volume(p, h), 1e-3)
        dC = (-q1 * C + q2 * (CB - C)) / V - kd * C
        if limites and h >= p.H and dh > 0.0:
            dh = 0.0          # transborda: o excedente sai com a mesma C
        return dh, dC

    dhC, dCc = mistura(hC, Cc, q1C, q2C, vC1, d["hj_C"], d["kd_C"], d["CB"])
    dhD, dCd = mistura(hD, Cd, q1D, q2D, vD1, d["hj_D"], d["kd_D"], d["CB"])

    if limites:
        if hA >= p.HA_max and dhA > 0.0:
            dhA = 0.0
        if hB >= p.HB_max and dhB > 0.0:
            dhB = 0.0
    return [dhA, dhB, dhC, dCc, dhD, dCd]


def _rk4(p, x, u, d, dt, limites):
    k1 = derivadas(p, x, u, d, limites)
    k2 = derivadas(p, [xi + dt / 2 * ki for xi, ki in zip(x, k1)], u, d, limites)
    k3 = derivadas(p, [xi + dt / 2 * ki for xi, ki in zip(x, k2)], u, d, limites)
    k4 = derivadas(p, [xi + dt * ki for xi, ki in zip(x, k3)], u, d, limites)
    return [xi + dt / 6 * (a + 2 * b + 2 * c + e)
            for xi, a, b, c, e in zip(x, k1, k2, k3, k4)]


def _saturar_estados(p, x):
    x[HA] = min(max(x[HA], 0.0), p.HA_max)
    x[HB] = min(max(x[HB], 0.0), p.HB_max)
    x[HC] = min(max(x[HC], 0.0), p.H)
    x[HD] = min(max(x[HD], 0.0), p.H)
    x[CC] = max(x[CC], 0.0)
    x[CD] = max(x[CD], 0.0)
    return x


# --------------------------------------------------------------- analisador
class Analisador:
    """Analisador de cloro: atraso de transporte theta seguido de primeira
    ordem tau_m, G_m(s) = exp(-theta s)/(tau_m s + 1), discretizado no passo dt."""

    def __init__(self, theta, tau_m, dt, C0):
        n = int(round(theta / dt))
        self.fila = deque([C0] * n, maxlen=n) if n > 0 else None
        self.a = math.exp(-dt / tau_m) if tau_m > 0 else 0.0
        self.y = C0

    def passo(self, C):
        if self.fila is not None:
            atrasado = self.fila[0]
            self.fila.append(C)
        else:
            atrasado = C
        self.y = self.a * self.y + (1 - self.a) * atrasado
        return self.y


# --------------------------------------------------------- laco principal
def simular(p, t_final, controlador, x0=None, u0=None, perturbacao=None,
            Ts=30.0, dt=1.0, limites=True):
    """Simula a planta por t_final segundos.

    controlador(t, medidas, u_anterior) -> u (8 comandos). E chamado a cada
    Ts segundos; entre chamadas o comando fica constante (segurador de ordem
    zero). As medidas sao os niveis (ideais) e as concentracoes vistas pelos
    analisadores: chaves 'hA', 'hB', 'hC', 'hD', 'CmC', 'CmD'.

    perturbacao(t) -> dict com as chaves que mudam em relacao ao nominal
    ('kd_C', 'kd_D', 'CB', 'hj_C', 'hj_D').
    """
    xe, ue = ponto_operacao(p)
    x = list(xe if x0 is None else x0)
    u = np.array(ue if u0 is None else u0, dtype=float)
    dnom = perturbacoes_nominais(p)

    N = int(round(t_final / dt))
    m = max(int(round(Ts / dt)), 1)
    anaC = Analisador(p.theta, p.tau_m, dt, x[CC])
    anaD = Analisador(p.theta, p.tau_m, dt, x[CD])

    T = np.arange(N + 1) * dt
    X = np.zeros((N + 1, 6))
    U = np.zeros((N + 1, 8))
    Cm = np.zeros((N + 1, 2))

    for k in range(N + 1):
        t = k * dt
        d = dict(dnom)
        if perturbacao is not None:
            d.update(perturbacao(t))
        if k % m == 0:
            med = {"hA": x[HA], "hB": x[HB], "hC": x[HC], "hD": x[HD],
                   "CmC": anaC.y, "CmD": anaD.y}
            u = np.clip(np.asarray(controlador(t, med, u.copy()), float), 0.0, 1.0)
        X[k], U[k], Cm[k] = x, u, (anaC.y, anaD.y)
        if k == N:
            break
        x = _rk4(p, x, u, d, dt, limites)
        if limites:
            x = _saturar_estados(p, x)
        anaC.passo(x[CC])
        anaD.passo(x[CD])

    return {"t": T, "x": X, "u": U, "Cm": Cm}


def malha_aberta(t, med, u):
    """Controlador nulo: mantem o comando."""
    return u
