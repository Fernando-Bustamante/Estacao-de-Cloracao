"""
Controladores da estacao de cloracao (TCPI).

Parte 2: PI digital com anti-windup e malhas de nivel dos reservatorios A e B
(PI + feedforward do consumo).
"""
import math
import simulador as s


def medida_valida(Cm, faixa=(0.0, 20.0)):
    """Sinal do analisador utilizavel: finito (NaN representa a quebra do
    sinal 4-20 mA) e dentro da faixa do instrumento."""
    return Cm is not None and math.isfinite(Cm) and faixa[0] <= Cm <= faixa[1]


class PI:
    """PI digital (forma posicional, integral por Euler) com anti-windup por
    recalculo (back-calculation). A saida e  u = u_ff + Kc*e + I, saturada em
    [umin, umax]. O integrador recebe Kc*Ts/Ti*e + Ts/Tt*(u_sat - u_calc)."""

    def __init__(self, Kc, Ti, Ts, Tt=None, umin=0.0, umax=1.0, I0=0.0):
        self.Kc, self.Ti, self.Ts = Kc, Ti, Ts
        self.Tt = Ti if Tt is None else Tt
        self.umin, self.umax = umin, umax
        self.I = I0

    def passo(self, ref, y, u_ff=0.0):
        e = ref - y
        u_calc = u_ff + self.Kc * e + self.I
        u_sat = min(max(u_calc, self.umin), self.umax)
        self.I += self.Kc * self.Ts / self.Ti * e + self.Ts / self.Tt * (u_sat - u_calc)
        return u_sat


# ------------------------------------------------ Parte 2: reservatorios
def sintonia_reservatorio(k_linha, delta_d, delta_h_max, Ts):
    """Sintonia SIMC para integrador com tau_c escolhido pelo R8.

    k_linha     ganho do integrador visto pelo PI (m/s por unidade de comando)
    delta_d     maior degrau de consumo, em unidades do comando da valvula
    delta_h_max desvio maximo de nivel admitido (R8)
    Retorna dicionario com tau_c, theta, Kc, Ti e o pico previsto sem feedforward.
    """
    theta = Ts / 2                                  # atraso equivalente do ZOH
    t_deriva = delta_h_max / (k_linha * delta_d)    # tempo para derivar delta_h_max
    tau_c = t_deriva - theta
    Kc = 1.0 / (k_linha * (tau_c + theta))          # = delta_d / delta_h_max
    Ti = 4.0 * (tau_c + theta)
    pico = 2.0 * math.exp(-1.0) * k_linha * delta_d * (tau_c + theta)
    return {"tau_c": tau_c, "theta": theta, "Kc": Kc, "Ti": Ti,
            "t_deriva": t_deriva, "pico_sem_ff": pico}


class ControleReservatorios:
    """LIC-A e LIC-B: PI com feedforward do consumo (Secao 4, reservatorios).

    v_A1 = PI_A(hA) + gA*(pAC + pAD),   gA = K1/KA1 = 0,5
    v_B1 = PI_B(hB) + gB*(pBC + pBD),   gB = K2/KB1 = 0,5
    fator_ff permite testar o feedforward desligado (0) ou com erro de ganho.
    """

    def __init__(self, p, sint_A, sint_B, Ts=30.0, fator_ff=1.0):
        self.p = p
        self.gA = p.K1 / p.KA1 * fator_ff
        self.gB = p.K2 / p.KB1 * fator_ff
        self.refA, self.refB = p.hA_nom, p.hB_nom
        _, u0 = s.ponto_operacao(p)
        # integrador inicial = parcela do comando nominal que o FF nao fornece
        I0A = u0[s.VA1] - self.gA * (u0[s.PAC] + u0[s.PAD])
        I0B = u0[s.VB1] - self.gB * (u0[s.PBC] + u0[s.PBD])
        self.piA = PI(sint_A["Kc"], sint_A["Ti"], Ts, I0=I0A)
        self.piB = PI(sint_B["Kc"], sint_B["Ti"], Ts, I0=I0B)

    def atualizar(self, med, u):
        """Recebe as medidas e o vetor de comandos ja com as bombas definidas
        neste ciclo; preenche vA1 e vB1."""
        u[s.VA1] = self.piA.passo(self.refA, med["hA"], self.gA * (u[s.PAC] + u[s.PAD]))
        u[s.VB1] = self.piB.passo(self.refB, med["hB"], self.gB * (u[s.PBC] + u[s.PBD]))
        return u


# ------------------------------------- Parte 3: nivel dos tanques C e D
def sintonia_nivel_imc(p, tau_c, Ts, h=None):
    """PI pelo metodo IMC para G_h3(s) = K3/(tau_h s + 1), com o atraso
    equivalente da amostragem theta = Ts/2.  Kc = tau_h/(K3 (tau_c+theta)),
    Ti = tau_h (o zero do PI cancela o polo do processo)."""
    h = p.h_nom if h is None else h
    A_ = s.area(p, h)
    u3 = p.q3_nom / (p.Kv * math.sqrt(h))
    tau_h = 2 * h * A_ / p.q3_nom
    K3 = -2 * h / u3
    theta = Ts / 2
    Kc = tau_h / (K3 * (tau_c + theta))
    return {"tau_c": tau_c, "theta": theta, "tau_h": tau_h, "K3": K3, "Kc": Kc, "Ti": tau_h}


class ControleNivelMistura:
    """LIC de um tanque de mistura: PI na valvula de descarga (h <- u3),
    ganho negativo (acao direta: se o nivel sobe, a valvula abre)."""

    def __init__(self, p, sint, Ts=30.0, tanque="C"):
        _, u0 = s.ponto_operacao(p)
        self.iu = s.VC1 if tanque == "C" else s.VD1
        self.ih = "hC" if tanque == "C" else "hD"
        self.ref = p.h_nom
        self.pi = PI(sint["Kc"], sint["Ti"], Ts, I0=u0[self.iu])

    def atualizar(self, med, u):
        u[self.iu] = self.pi.passo(self.ref, med[self.ih])
        return u


# ------------------------------------------ Parte 4: cloro (AIC + razao)
def dose_nominal(p):
    """Dose D = q2*CB/q1 no ponto nominal (Eq. da dose, Secao 4)."""
    _, u0 = s.ponto_operacao(p)
    return p.K2 * u0[s.PBC] * p.CB / (p.K1 * u0[s.PAC])


def sintonia_cloro_imc(p, tau_c, Ts):
    """PI IMC para a malha dose -> concentracao medida.

    Com a estacao de razao, du2 = (u2/D) dD, e a planta vista pelo AIC e
    G_D(s) = (u2/D) G_C2(s) = K_D/(tau_C s + 1), K_D = C/D. O analisador
    (atraso 60 s e lag 30 s) e a amostragem (Ts/2) sao aproximados pela regra
    da metade: theta = theta_a + tau_m/2 + Ts/2 e tau = tau_C + tau_m/2."""
    _, u0 = s.ponto_operacao(p)
    V = s.volume(p, p.h_nom)
    tau_C = 1.0 / (p.q3_nom / V + p.kd)
    D = dose_nominal(p)
    K_D = p.C_nom / D
    theta = p.theta + p.tau_m / 2 + Ts / 2
    tau = tau_C + p.tau_m / 2
    Kc = tau / (K_D * (tau_c + theta))
    # maior dose necessaria: kd = 3 1/h e CB 20% abaixo do nominal
    D_max = p.C_nom * (1 + (3 / 3600) * V / p.q3_nom) / 0.8
    return {"tau_c": tau_c, "theta": theta, "tau": tau, "tau_C": tau_C,
            "K_D": K_D, "D": D, "D_max": D_max, "Kc": Kc, "Ti": tau}


class ControleCloro:
    """AIC (PI na dose, ganho escalonado por V(h)/V_nom, integracao
    condicional) + FY (razao u2 = k u1 D, k = K1/(K2 CB_nom)).

    Anti-windup: a integral para quando u2 satura, quando a dose atinge seus
    limites [0, D_max] ou quando nao ha vazao de agua (u1 < 5% do nominal),
    caso em que u2 = k u1 D fica preso em zero sem nunca saturar.

    Falha do analisador (medida invalida): o AIC passa a manual com a ultima
    dose, e a razao continua dosando proporcionalmente a vazao. Quando a
    medida volta, a integral e recalculada para retomar sem salto."""

    def __init__(self, p, sint, Ts=30.0, tanque="C", escalonar=True):
        _, u0 = s.ponto_operacao(p)
        self.p, self.Ts, self.escalonar = p, Ts, escalonar
        self.Kc, self.Ti = sint["Kc"], sint["Ti"]
        self.k = p.K1 / (p.K2 * p.CB)
        self.Vnom = s.volume(p, p.h_nom)
        self.ref = p.C_nom
        self.I = sint["D"]
        self.D_max = sint["D_max"]
        self.i1, self.i2 = (s.PAC, s.PBC) if tanque == "C" else (s.PAD, s.PBD)
        self.ih, self.ic = ("hC", "CmC") if tanque == "C" else ("hD", "CmD")
        self.u1_min = 0.05 * u0[self.i1]
        self.D = sint["D"]
        self.manual = False

    def fator(self, h):
        return s.volume(self.p, h) / self.Vnom if self.escalonar else 1.0

    def atualizar(self, med, u):
        if not medida_valida(med[self.ic]):
            self.manual = True                     # AIC em manual: mantem a dose
            u[self.i2] = min(max(self.k * u[self.i1] * self.D, 0.0), 1.0)
            return u
        e = self.ref - med[self.ic]
        Kc = self.Kc * self.fator(med[self.ih])
        if self.manual:                            # volta ao automatico sem salto
            self.I, self.manual = self.D - Kc * e, False
        D_calc = Kc * e + self.I
        D = min(max(D_calc, 0.0), self.D_max)
        u2_calc = self.k * u[self.i1] * D
        u2 = min(max(u2_calc, 0.0), 1.0)
        trava = ((u2_calc > 1.0 or D_calc > self.D_max) and e > 0) or \
                ((u2_calc < 0.0 or D_calc < 0.0) and e < 0) or u[self.i1] < self.u1_min
        if not trava:
            self.I += Kc * self.Ts / self.Ti * e
        self.D = D
        u[self.i2] = u2
        return u


# ---------------------------------- Parte 5: override de nivel alto
def sintonia_override(p, tau_c, Ts, h_lim=2.7):
    """LC-HH: SIMC para integrador, com o ganho de alta frequencia da bomba de
    agua sobre o nivel perto do limite, k' = K1/A(h_lim).

    A referencia fica abaixo do limite do R6 por uma folga igual ao maior
    excesso previsto no pior caso (valvula de descarga fechada, entrada
    inteira sem saida: delta_d = u1 nominal):
        excesso = 2 e^-1 k' delta_d (tau_c + theta)."""
    _, u0 = s.ponto_operacao(p)
    k = p.K1 / s.area(p, h_lim)
    theta = Ts / 2
    excesso = 2.0 * math.exp(-1.0) * k * u0[s.PAC] * (tau_c + theta)
    h_ref = math.floor(100 * (h_lim - excesso)) / 100
    return {"tau_c": tau_c, "theta": theta, "k": k, "h_lim": h_lim,
            "excesso_previsto": excesso, "h_max": h_ref,
            "Kc": 1.0 / (k * (tau_c + theta)), "Ti": 4.0 * (tau_c + theta)}


class Override:
    """LC-HH (PI com realimentacao externa de reset) + seletor de minimo LS.

    u_HH = Kc*(h_max - h) + F, com F = saida selecionada filtrada por
    1/(Ti s + 1). Fora de uso, F acompanha u1_prod e o integrador nao
    enrola; quando o nivel passa de h_max, u_HH fica menor que u1_prod e o
    seletor passa o comando da bomba de agua ao LC-HH, sem salto."""

    def __init__(self, p, sint, Ts=30.0, tanque="C", ativo=True):
        _, u0 = s.ponto_operacao(p)
        self.Kc, self.Ti, self.Ts = sint["Kc"], sint["Ti"], Ts
        self.h_max, self.ativo = sint["h_max"], ativo
        self.ih = "hC" if tanque == "C" else "hD"
        self.F = u0[s.PAC]
        self.selecionado = False

    def selecionar(self, med, u1_prod):
        if not self.ativo:
            return u1_prod
        u_hh = self.Kc * (self.h_max - med[self.ih]) + self.F
        u1 = min(max(min(u1_prod, u_hh), 0.0), 1.0)
        self.selecionado = u_hh < u1_prod
        self.F += self.Ts / self.Ti * (u1 - self.F)
        return u1


# ------------------------------- estrutura de contingencia completa
def sintonias_padrao(p, Ts=30.0, tau_c_override=150.0):
    """Sintonias escolhidas nas partes 2 a 5."""
    _, u0 = s.ponto_operacao(p)
    return {
        "A": sintonia_reservatorio(p.KA1 / p.AA, 0.2 * p.K1 / p.KA1 * 2 * u0[s.PAC], 0.10, Ts),
        "B": sintonia_reservatorio(p.KB1 / p.AB, 0.2 * p.K2 / p.KB1 * 2 * u0[s.PBC], 0.10, Ts),
        "nivel": sintonia_nivel_imc(p, 600.0, Ts),
        "cloro": sintonia_cloro_imc(p, 150.0, Ts),
        "override": sintonia_override(p, tau_c_override, Ts),
    }


class Contingencia:
    """LIC + AIC/FY + override (LC-HH/LS) nos tanques C e D e LIC-A/LIC-B
    com feedforward nos reservatorios. Perfis sao funcoes do tempo:
    Cref(t), href(t) e prod(t) (fator sobre a producao nominal)."""

    def __init__(self, p, Ts=30.0, sint=None, Cref=None, href=None, prod=None,
                 override=True, escalonar=True):
        self.p = p
        self.sint = sintonias_padrao(p, Ts) if sint is None else sint
        _, self.u0 = s.ponto_operacao(p)
        self.Cref = Cref or (lambda t: p.C_nom)
        self.href = href or (lambda t: p.h_nom)
        self.prod = prod or (lambda t: 1.0)
        self.resv = ControleReservatorios(p, self.sint["A"], self.sint["B"], Ts)
        self.lic = {k: ControleNivelMistura(p, self.sint["nivel"], Ts, k) for k in "CD"}
        self.aic = {k: ControleCloro(p, self.sint["cloro"], Ts, k, escalonar) for k in "CD"}
        self.ovr = {k: Override(p, self.sint["override"], Ts, k, override) for k in "CD"}

    def tanque(self, k, t, med, u):
        """Um ciclo de um tanque: seletor/override (u1) -> LIC (u3) -> AIC/FY (u2)."""
        i1 = s.PAC if k == "C" else s.PAD
        u[i1] = self.ovr[k].selecionar(med, self.u0[i1] * self.prod(t))
        self.lic[k].ref = self.href(t)
        self.aic[k].ref = self.Cref(t)
        u = self.lic[k].atualizar(med, u)
        return self.aic[k].atualizar(med, u)

    def inicializar(self, k, t, med, u):
        """Troca sem salto: os integradores do tanque k partem dos comandos
        que estao sendo aplicados (u), para que a primeira saida da
        contingencia repita esses comandos."""
        lic, aic, ovr = self.lic[k], self.aic[k], self.ovr[k]
        lic.ref, aic.ref = self.href(t), self.Cref(t)
        lic.pi.I = u[lic.iu] - lic.pi.Kc * (lic.ref - med[lic.ih])
        u1 = u[aic.i1]
        D = u[aic.i2] / (aic.k * u1) if u1 > aic.u1_min else aic.D
        aic.D = min(max(D, 0.0), aic.D_max)
        if medida_valida(med[aic.ic]):
            aic.I = aic.D - aic.Kc * aic.fator(med[aic.ih]) * (aic.ref - med[aic.ic])
            aic.manual = False
        else:
            aic.I, aic.manual = aic.D, True
        ovr.F = u1

    def __call__(self, t, med, u):
        u = u.copy()
        for k in "CD":
            u = self.tanque(k, t, med, u)
        return self.resv.atualizar(med, u)
