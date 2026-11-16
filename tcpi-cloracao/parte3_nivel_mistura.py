"""
Parte 3: sintonia e teste da malha de nivel dos tanques de mistura (h <- u3).

PI pelo metodo IMC no ponto nominal, sintonia fixa (Secao 4.4: o ganho de alta
frequencia do nivel varia so +-10% na faixa de operacao).
Testes no modelo nao linear: degraus em h_ref (R4 e R7) e mudanca de producao
de +-20% (R5). Os reservatorios usam as malhas da parte 2; a dosagem segue a
vazao de agua na razao nominal (a malha de cloro entra na parte 4).
"""
import math
import numpy as np
import control as ct
import warnings
import simulador as s
import controladores as c

warnings.filterwarnings("ignore")
p = s.Parametros()
x0, u0 = s.ponto_operacao(p)
Ts = 30.0
TAU_C = 600.0                    # 10 min

# ------------------------------------------------------------- sintonia
sn = c.sintonia_nivel_imc(p, TAU_C, Ts)
print(f"Sintonia IMC: tau_h = {sn['tau_h']:.0f} s ({sn['tau_h']/60:.1f} min), K3 = {sn['K3']:.3f}")
print(f"  tau_c = {TAU_C:.0f} s, theta = {sn['theta']:.0f} s ->  Kc = {sn['Kc']:.3f} /m,"
      f"  Ti = {sn['Ti']:.0f} s ({sn['Ti']/60:.1f} min)")

# ------------------------------------ robustez no modelo linear (R7)
sv = ct.tf("s")
pade = ct.tf(*ct.pade(sn["theta"], 3))
C_pi = sn["Kc"] * (1 + 1 / (sn["Ti"] * sv))
print("\nSintonia fixa avaliada no modelo linear em outros niveis (R7)")
print("  h_ref   tau_h     K3      MG     MF    ts(2%)   Mp")
for h in (1.5, 2.0, 2.5):
    si = c.sintonia_nivel_imc(p, TAU_C, Ts, h)
    G = si["K3"] / (si["tau_h"] * sv + 1) * pade
    L = C_pi * G
    m = ct.stability_margins(L)
    T = ct.minreal(L / (1 + L), verbose=False)
    t = np.linspace(0, 4 * 3600, 8001)
    _, y = ct.step_response(T, t)
    inf = ct.step_info(y, t, SettlingTimeThreshold=0.02)
    print(f"  {h:.1f}   {si['tau_h']/60:5.1f}min  {si['K3']:6.2f}  {m[0]:5.1f}  {m[1]:5.1f}"
          f"  {inf['SettlingTime']/60:5.1f}min  {inf['Overshoot']:4.1f}%")

# ------------------------------------------ controlador para os testes
def montar(perfil_ref, perfil_prod):
    resv = c.ControleReservatorios(
        p,
        c.sintonia_reservatorio(p.KA1 / p.AA, 0.2 * p.K1 / p.KA1 * 2 * u0[s.PAC], 0.10, Ts),
        c.sintonia_reservatorio(p.KB1 / p.AB, 0.2 * p.K2 / p.KB1 * 2 * u0[s.PBC], 0.10, Ts), Ts)
    licC = c.ControleNivelMistura(p, sn, Ts, "C")
    licD = c.ControleNivelMistura(p, sn, Ts, "D")

    def f(t, med, u):
        u = u.copy()
        fator = perfil_prod(t)
        for ip, ib in ((s.PAC, s.PBC), (s.PAD, s.PBD)):
            u[ip] = u0[ip] * fator
            u[ib] = u0[ib] * fator          # dosagem em razao com a agua
        licC.ref = licD.ref = perfil_ref(t)
        u = licC.atualizar(med, u)
        u = licD.atualizar(med, u)
        return resv.atualizar(med, u)
    return f


def degraus(lista):
    """lista de (instante, valor) -> funcao degrau por partes."""
    def f(t):
        v = lista[0][1]
        for ti, vi in lista:
            if t >= ti:
                v = vi
        return v
    return f


def metricas(r, t_ini, t_fim, r0, r1):
    t, h = r["t"], r["x"][:, s.HC]
    jan = (t >= t_ini) & (t < t_fim)
    tt, hh = t[jan], h[jan]
    passo = r1 - r0
    fora = np.where(np.abs(hh - r1) > 0.02 * abs(passo))[0]
    ts = (tt[fora[-1]] - t_ini) if len(fora) else 0.0
    mp = 100 * max(0.0, np.max((hh - r1) * np.sign(passo))) / abs(passo)
    u3 = r["u"][jan, s.VC1]
    sat = np.sum((u3 <= 0.0) | (u3 >= 1.0)) * (t[1] - t[0])
    return ts, mp, u3.min(), u3.max(), sat

# --------------------------------- R4 e R7: sequencia de degraus em h_ref
H = 3600.0
seq = [(0, 2.0), (600, 2.25), (5 * H, 2.0), (10 * H, 1.5), (15 * H, 1.75), (20 * H, 2.5), (25 * H, 2.25)]
r = s.simular(p, 30 * H, montar(degraus(seq), lambda t: 1.0), Ts=Ts)
print("\nDegraus em h_ref no modelo nao linear (R4: ts <= 60 min, Mp <= 10%)")
print("  degrau          ts(2%)    Mp     u3 min-max   tempo saturado")
for k in range(1, len(seq)):
    t_ini, r1 = seq[k]
    r0 = seq[k - 1][1]
    t_fim = seq[k + 1][0] if k + 1 < len(seq) else 30 * H
    ts, mp, umin, umax, sat = metricas(r, t_ini, t_fim, r0, r1)
    print(f"  {r0:.2f} -> {r1:.2f} m   {ts/60:5.1f} min  {mp:4.1f}%   {umin:.2f}-{umax:.2f}     {sat/60:4.1f} min")
print(f"  nivel minimo e maximo em toda a sequencia: {r['x'][:, s.HC].min():.3f} a {r['x'][:, s.HC].max():.3f} m")

# ------------------------------------------- R5: mudanca de producao
print("\nMudanca de producao de +-20% em t = 10 min (R5: |dh| <= 0,10 m)")
for sinal in (+1, -1):
    r = s.simular(p, 8 * H, montar(lambda t: 2.0, degraus([(0, 1.0), (600, 1 + 0.2 * sinal)])), Ts=Ts)
    dh = r["x"][:, s.HC] - 2.0
    u3 = r["u"][:, s.VC1]
    i = np.argmax(np.abs(dh))
    print(f"  {sinal*20:+d}%: |dh| max = {abs(dh[i]):.3f} m em t = {r['t'][i]/60:.0f} min,"
          f" dh em 8 h = {dh[-1]:+.4f} m, u3 de {u3.min():.3f} a {u3.max():.3f},"
          f" |dhA| max = {100*np.abs(r['x'][:, s.HA]-p.hA_nom).max():.2f} cm")
