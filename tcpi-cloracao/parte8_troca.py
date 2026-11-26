"""
Parte 8: troca automatica MPC -> contingencia (Secao 4 do relatorio).

Se o QP de um tanque falha ou se o sinal do analisador fica invalido, o
tanque passa a contingencia no mesmo instante de amostragem, com os
integradores inicializados pelos comandos aplicados (sem salto). Testes no
tanque C, no modelo nao linear:
  1. sem falha: a troca automatica nao muda nada;
  2. falha do otimizador em 60 min e degrau de k_d de 1 para 2 1/h em 90 min,
     com e sem a troca (sem a troca, o MPC so mantem o ultimo comando);
  3. sinal do analisador invalido (NaN) de 60 a 180 min e producao +20% em
     90 min: o AIC fica em manual com a ultima dose e volta sem salto.
"""
import math
import numpy as np
import simulador as s
import mpc as M

p = s.Parametros()
Ts, H = 30.0, 3600.0
T_FALHA, T_PERT, T_VOLTA = 60 * 60.0, 90 * 60.0, 180 * 60.0
x0, u0 = s.ponto_operacao(p)
IU = {"u1": s.PAC, "u2": s.PBC, "u3": s.VC1}


def degraus(lista):
    def f(t):
        v = lista[0][1]
        for ti, vi in lista:
            if t >= ti:
                v = vi
        return v
    return f


def acomodacao(t, y, t_ini, alvo, faixa):
    j = t >= t_ini
    fora = np.where(np.abs(y[j] - alvo) > faixa)[0]
    return (t[j][fora[-1]] - t_ini) if len(fora) else 0.0


def salto(r, t_ev):
    """Maior variacao dos comandos do tanque C no instante t_ev, comparada a
    maior variacao entre duas amostras quaisquer do MPC antes da falha."""
    t, u = r["t"], r["u"]
    k = int(np.searchsorted(t, t_ev))
    return {n: u[k, i] - u[k - 1, i] for n, i in IU.items()}


def txt_salto(d):
    return ", ".join(f"{n} {v:+.4f}" for n, v in d.items())


class FalhaQP:
    """Falha do otimizador do tanque k a partir de t_ini (MPC.falha_simulada):
    o passo devolve o comando anterior com solver_ok = False, como quando o
    OSQP nao converge."""

    def __init__(self, ctl, k, t_ini):
        self.ctl, self.m, self.t_ini = ctl, ctl.mpc[k], t_ini

    def __call__(self, t, med, u):
        self.m.falha_simulada = t >= self.t_ini
        return self.ctl(t, med, u)


class FalhaAnalisador:
    """Sinal do analisador invalido (NaN) entre t_ini e t_fim."""

    def __init__(self, ctl, chave, t_ini, t_fim):
        self.ctl, self.chave, self.t_ini, self.t_fim = ctl, chave, t_ini, t_fim

    def __call__(self, t, med, u):
        if self.t_ini <= t < self.t_fim:
            med = {**med, self.chave: math.nan}
        return self.ctl(t, med, u)


kd2 = lambda t: {"kd_C": 2 / 3600} if t >= T_PERT else {}

# 1. sem falha
print("1. Sem falha (degrau de k_d em 90 min, 4 h)")
ra = s.simular(p, 4 * H, M.ControleMPC(p, Ts), perturbacao=kd2, Ts=Ts)
cb = M.ControleMPC(p, Ts, troca_automatica=False)
rb = s.simular(p, 4 * H, cb, perturbacao=kd2, Ts=Ts)
print(f"   maior diferenca com e sem troca automatica: x {np.abs(ra['x'] - rb['x']).max():.1e},"
      f" u {np.abs(ra['u'] - rb['u']).max():.1e}")

# passo tipico do MPC, para comparar com o salto na troca
du = np.abs(np.diff(ra["u"][:, [s.PAC, s.PBC, s.VC1]], axis=0)).max(axis=0)
print(f"   maior movimento do MPC entre amostras: u1 {du[0]:.4f}, u2 {du[1]:.4f}, u3 {du[2]:.4f}")

# 2. falha do otimizador
print("\n2. Falha do otimizador do tanque C em 60 min; k_d de 1 para 2 1/h em 90 min (6 h)")
for troca in (True, False):
    ctl = M.ControleMPC(p, Ts, troca_automatica=troca)
    r = s.simular(p, 6 * H, FalhaQP(ctl, "C", T_FALHA), perturbacao=kd2, Ts=Ts)
    t, h, C = r["t"], r["x"][:, s.HC], r["x"][:, s.CC]
    dep = t >= T_PERT
    print(f"   troca {'sim' if troca else 'nao'}: eventos {[(round(e[0] / 60), e[1], e[2]) for e in getattr(ctl, 'eventos', [])]},"
          f" modo final {ctl.modo}")
    if troca:
        print(f"     salto na troca: {txt_salto(salto(r, T_FALHA))}")
    print(f"     C apos o degrau: {C[dep].min():.3f} a {C[dep].max():.3f} mg/L,"
          f" |dC| max {np.abs(C[dep] - 2).max():.3f}, volta a +-2% em {acomodacao(t, C, T_PERT, 2.0, 0.04) / 60:.1f} min,"
          f" C final {C[-1]:.3f}")
    print(f"     h: {h.min():.3f} a {h.max():.3f} m;  tanque D: C {r['x'][:, s.CD].min():.3f}-{r['x'][:, s.CD].max():.3f}")

# 3. falha do analisador
print("\n3. Analisador do tanque C invalido de 60 a 180 min; producao +20% em 90 min (6 h)")
ctl = M.ControleMPC(p, Ts, prod=degraus([(0, 1.0), (T_PERT, 1.2)]))
r = s.simular(p, 6 * H, FalhaAnalisador(ctl, "CmC", T_FALHA, T_VOLTA), Ts=Ts)
t, h, C = r["t"], r["x"][:, s.HC], r["x"][:, s.CC]
aic = ctl.cont.aic["C"]
print(f"   eventos {[(round(e[0] / 60), e[1], e[2]) for e in ctl.eventos]}, modo final {ctl.modo},"
      f" AIC em manual no fim: {aic.manual}")
print(f"   salto na troca (60 min): {txt_salto(salto(r, T_FALHA))}")
print(f"   salto na volta do analisador (180 min): {txt_salto(salto(r, T_VOLTA))}")
fal = (t >= T_FALHA) & (t < T_VOLTA)
print(f"   C durante a falha: {C[fal].min():.3f} a {C[fal].max():.3f} mg/L; depois: {C[t >= T_VOLTA].min():.3f} a"
      f" {C[t >= T_VOLTA].max():.3f}; final {C[-1]:.3f}")
print(f"   h: {h.min():.3f} a {h.max():.3f} m;  tanque D (MPC): C {r['x'][:, s.CD].min():.3f}-{r['x'][:, s.CD].max():.3f},"
      f" h {r['x'][:, s.HD].min():.3f}-{r['x'][:, s.HD].max():.3f}")
