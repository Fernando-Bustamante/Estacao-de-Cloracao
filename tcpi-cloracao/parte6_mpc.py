"""
Parte 6: MPC adaptativo nos tanques de mistura. Mesma bateria de testes das
partes 3 a 5, no modelo nao linear.
"""
import numpy as np
import simulador as s
import controladores as c
import mpc as M

p = s.Parametros()
Ts, H, t0 = 30.0, 3600.0, 600.0
x0, u0 = s.ponto_operacao(p)


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


def iniciar_em(h):
    q = p.__class__(**{**p.__dict__, "h_nom": h})
    return s.ponto_operacao(q)


def resumo_u(r):
    return (f"u1 {r['u'][:, s.PAC].min():.2f}-{r['u'][:, s.PAC].max():.2f}, "
            f"u2 {r['u'][:, s.PBC].min():.2f}-{r['u'][:, s.PBC].max():.2f}, "
            f"u3 {r['u'][:, s.VC1].min():.2f}-{r['u'][:, s.VC1].max():.2f}")


print("MPC adaptativo: Np = 120, Nc = 10, pesos de Bryson, Kalman")

# R2
for C1 in (2.2, 1.8):
    r = s.simular(p, 3 * H, M.ControleMPC(p, Ts, Cref=degraus([(0, 2.0), (t0, C1)])), Ts=Ts)
    C, t = r["x"][:, s.CC], r["t"]
    ts = acomodacao(t, C, t0, C1, 0.02 * abs(C1 - 2.0))
    mp = 100 * max(0.0, np.max((C[t >= t0] - C1) * np.sign(C1 - 2.0))) / abs(C1 - 2.0)
    print(f"  R2  C_ref 2,0 -> {C1}: ts = {ts/60:.1f} min, Mp = {mp:.1f}%, |dh| = {np.abs(r['x'][:, s.HC]-2).max():.3f} m; {resumo_u(r)}")

# R3
pert = [("kd 1 -> 2 1/h", lambda t: {"kd_C": (2 if t >= t0 else 1) / 3600}),
        ("kd 1 -> 0,2 1/h", lambda t: {"kd_C": (0.2 if t >= t0 else 1) / 3600}),
        ("CB -20%", lambda t: {"CB": p.CB * (0.8 if t >= t0 else 1)}),
        ("hj 0 -> 0,3 m", lambda t: {"hj_C": 0.3 if t >= t0 else 0.0})]
for nome, fp in pert:
    r = s.simular(p, 6 * H, M.ControleMPC(p, Ts), perturbacao=fp, Ts=Ts)
    C, t = r["x"][:, s.CC], r["t"]
    print(f"  R3  {nome:16s}: |dC| max = {np.abs(C-2).max():.3f} mg/L, volta a +-2% em"
          f" {acomodacao(t, C, t0, 2.0, 0.04)/60:5.1f} min, C final = {C[-1]:.4f},"
          f" |dh| max = {np.abs(r['x'][:, s.HC]-2).max():.3f} m, h final = {r['x'][-1, s.HC]:.4f}")

# R4
seq = [(0, 2.0), (t0, 2.25), (4 * H, 2.0), (8 * H, 1.5), (12 * H, 1.75), (16 * H, 2.5), (20 * H, 2.25)]
r = s.simular(p, 24 * H, M.ControleMPC(p, Ts, href=degraus(seq)), Ts=Ts)
t, h, C = r["t"], r["x"][:, s.HC], r["x"][:, s.CC]
print("  R4  degraus em h_ref:")
for k in range(1, len(seq)):
    t_ini, r1 = seq[k]; r0 = seq[k - 1][1]
    t_fim = seq[k + 1][0] if k + 1 < len(seq) else 24 * H
    jan = (t >= t_ini) & (t < t_fim)
    passo = r1 - r0
    ts = acomodacao(t[jan], h[jan], t_ini, r1, 0.02 * abs(passo))
    mp = 100 * max(0.0, np.max((h[jan] - r1) * np.sign(passo))) / abs(passo)
    print(f"      {r0:.2f} -> {r1:.2f} m: ts = {ts/60:5.1f} min, Mp = {mp:4.1f}%, |dC| = {np.abs(C[jan]-2).max():.3f} mg/L")
print(f"      nivel entre {h.min():.3f} e {h.max():.3f} m; C entre {C.min():.3f} e {C.max():.3f} mg/L")

# R5
for sinal in (+1, -1):
    r = s.simular(p, 8 * H, M.ControleMPC(p, Ts, prod=degraus([(0, 1.0), (t0, 1 + 0.2 * sinal)])), Ts=Ts)
    print(f"  R5  producao {sinal*20:+d}%: |dC| max = {np.abs(r['x'][:, s.CC]-2).max():.3f} mg/L,"
          f" |dh| max = {np.abs(r['x'][:, s.HC]-2).max():.3f} m, u1 final = {r['u'][-1, s.PAC]:.3f}")

# R6: cenarios de falha da parte 5
T_ON, T_OFF = 600.0, 4 * H
rede = lambda t: {"hj_C": 2.6 if T_ON <= t < T_OFF else 0.0}


class ValvulaTravada:
    def __init__(self, ctrl): self.ctrl = ctrl
    def __call__(self, t, med, u):
        u = self.ctrl(t, med, u)
        if T_ON <= t < T_OFF: u[s.VC1] = 0.0
        return u


for nome, env, fp in (("rede pressurizada (h_j = 2,6 m)", lambda c_: c_, rede),
                      ("valvula de descarga fechada", ValvulaTravada, None)):
    ctrl = env(M.ControleMPC(p, Ts))
    r = s.simular(p, 8 * H, ctrl, perturbacao=fp, Ts=Ts)
    t, h, C, u1 = r["t"], r["x"][:, s.HC], r["x"][:, s.CC], r["u"][:, s.PAC]
    comv = (t < T_ON) | (u1 > 0.05)
    print(f"  R6  {nome}: h max = {h.max():.3f} m, u1 min = {u1.min():.3f},"
          f" producao media no evento = {100*u1[(t>2*H)&(t<T_OFF)].mean()/u0[s.PAC]:.0f}%,"
          f" C (com vazao) = {C[comv].min():.2f}-{C[comv].max():.2f}, h em 8 h = {h[-1]:.3f}")

# R7: operacao em 1,5 e 2,5 m e atraso 2 theta, com degrau em kd
for h0 in (1.5, 2.5):
    for adapt in (True, False):
        xh, uh = iniciar_em(h0)
        r = s.simular(p, 6 * H, M.ControleMPC(p, Ts, href=lambda t, h0=h0: h0, adaptativo=adapt),
                      x0=xh, u0=uh, perturbacao=pert[0][1], Ts=Ts)
        C, t = r["x"][:, s.CC], r["t"]
        tail = C[t >= 5 * H]
        print(f"  R7  h = {h0} m, adaptativo {'sim' if adapt else 'nao'}: |dC| max = {np.abs(C-2).max():.3f},"
              f" volta em {acomodacao(t, C, t0, 2.0, 0.04)/60:5.1f} min, oscilacao final = {tail.max()-tail.min():.4f}")
q = p.__class__(**{**p.__dict__, "theta": 2 * p.theta})
r = s.simular(q, 6 * H, M.ControleMPC(p, Ts), perturbacao=pert[0][1], Ts=Ts)
C, t = r["x"][:, s.CC], r["t"]
print(f"  R7  atraso do analisador 2 theta (MPC com modelo nominal): |dC| max = {np.abs(C-2).max():.3f},"
      f" volta em {acomodacao(t, C, t0, 2.0, 0.04)/60:.1f} min, oscilacao final = {C[t>=5*H].max()-C[t>=5*H].min():.4f}")
