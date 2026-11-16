"""
Parte 1: validacao do simulador base em malha aberta.

Confere o simulador contra o que ja esta no relatorio:
  1. equilibrio nominal parado por 8 h
  2. Tabela 5 (desvio entre modelo nao linear e linearizado)
  3. precisao do passo fixo (RK4, dt = 1 s) contra integrador de referencia
  4. rampas dos reservatorios (integradores)
  5. analisador de cloro (atraso de 60 s + primeira ordem de 30 s)
"""
import math
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import simulador as s

p = s.Parametros()
x0, u0 = s.ponto_operacao(p)
T8 = 8 * 3600

print("Ponto de operacao")
print("  u0 =", np.round(u0, 4), " (relatorio: vA1 0,600; vB1 0,347; u1 0,600; u2 0,347; u3 0,500)")

# ------------------------------------------------------------ 1. equilibrio
t0 = time.time()
r = s.simular(p, T8, s.malha_aberta)
dur = time.time() - t0
print(f"\n1. Equilibrio por 8 h: maior desvio = {np.abs(r['x'] - x0).max():.2e}"
      f"   (tempo de simulacao: {dur:.1f} s)")

# ------------------------------------------------- 2. Tabela 5 do relatorio
def modelo_linear(p):
    h, C = p.h_nom, p.C_nom
    A_, V_ = s.area(p, h), s.volume(p, h)
    q3 = p.q3_nom
    A = np.array([[-q3 / (2 * h * A_), 0.0],
                  [-p.kd * C * A_ / V_, -(q3 / V_ + p.kd)]])
    B = np.array([[p.K1 / A_, p.K2 / A_, -p.Kv * math.sqrt(h) / A_],
                  [-p.K1 * C / V_, p.K2 * (p.CB - C) / V_, 0.0]])
    return A, B


def resposta_linear(A, B, du, t):
    """Resposta exata do modelo linear a um degrau du aplicado em t = 0."""
    n = len(t)
    M = np.zeros((4, 4))
    M[:2, :2], M[:2, 2:3] = A, (B @ du).reshape(2, 1)
    dt = t[1] - t[0]
    Phi = expm(M * dt)
    z = np.zeros(4)
    z[2] = 1.0
    out = np.zeros((n, 2))
    for k in range(1, n):
        z = Phi @ z
        out[k] = z[:2]
    return out


A, B = modelo_linear(p)
tabela5 = {  # valores publicados: -20, -10, -5, +5, +10, +20 %
    ("u1", "h"): [10.4, 4.7, 2.2, 2.0, 3.7, 6.4],
    ("u1", "C"): [22.1, 11.1, 5.5, 5.6, 11.2, 22.6],
    ("u2", "h"): [0.0] * 6,
    ("u2", "C"): [0.0] * 6,
    ("u3", "h"): [23.1, 12.9, 6.7, 7.1, 14.6, 30.0],
    ("u3", "C"): [15.1, 9.5, 5.2, 5.8, 12.2, 26.0],
}
passos = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]
entradas = {"u1": (s.PAC, 0), "u2": (s.PBC, 1), "u3": (s.VC1, 2)}

print("\n2. Tabela 5 (simulador / relatorio, em %)")
print("   entrada saida   " + "".join(f"{100*q:+8.0f}%     " for q in passos))
pior = 0.0
for nome, (iu, il) in entradas.items():
    res = {"h": [], "C": []}
    for q in passos:
        u = u0.copy()
        u[iu] *= 1 + q
        rr = s.simular(p, T8, lambda t, m, uu, u=u: u, limites=False, Ts=3600)
        du = np.zeros(3)
        du[il] = u0[iu] * q
        lin = resposta_linear(A, B, du, rr["t"])
        for j, (sx, nom) in enumerate([(s.HC, p.h_nom), (s.CC, p.C_nom)]):
            nl = rr["x"][:, sx] - nom
            var = np.abs(nl).max()
            res["h" if j == 0 else "C"].append(
                100 * np.abs(nl - lin[:, j]).max() / var if var > 1e-12 else 0.0)
    for saida in ("h", "C"):
        pub = tabela5[(nome, saida)]
        cel = "".join(f"{a:5.1f}/{b:<5.1f}  " for a, b in zip(res[saida], pub))
        print(f"   {nome:6s}  {saida:5s}  {cel}")
        if nome != "u2":
            pior = max(pior, max(abs(a - b) for a, b in zip(res[saida], pub)))
print(f"   maior diferenca para a tabela publicada: {pior:.2f} ponto percentual")

# ------------------------------------------- 3. precisao do passo fixo RK4
u = u0.copy()
u[s.VC1] *= 0.8
rr = s.simular(p, T8, lambda t, m, uu: u, limites=False, Ts=3600)
dn = s.perturbacoes_nominais(p)
ref = solve_ivp(lambda t, x: s.derivadas(p, x, u, dn, False), [0, T8], list(x0),
                t_eval=rr["t"], rtol=1e-10, atol=1e-12, method="DOP853")
erro = np.abs(rr["x"] - ref.y.T).max(axis=0)
print("\n3. RK4 (dt = 1 s) x integrador de referencia, degrau de -20% na valvula:")
print(f"   maior erro: h = {erro[s.HC]:.1e} m, C = {erro[s.CC]:.1e} mg/L;"
      f" nivel em 8 h = {rr['x'][-1, s.HC]:.3f} m (relatorio: 3,03 m)")

# ---------------------------------------------------- 4. reservatorios
u = u0.copy()
u[s.PAC] += 0.1
u[s.PBC] += 0.1
rr = s.simular(p, 3600, lambda t, m, uu: u)
dA = (rr["x"][-1, s.HA] - rr["x"][0, s.HA])
dB = (rr["x"][-1, s.HB] - rr["x"][0, s.HB])
print("\n4. Reservatorios, +0,1 no comando de PAC e de PBC por 1 h:")
print(f"   A: {dA:+.4f} m/h (relatorio: -0,2865)   B: {100*dB:+.4f} cm/h (relatorio: -0,2547)")

# ---------------------------------------------------------- 5. analisador
an = s.Analisador(p.theta, p.tau_m, 1.0, 0.0)
y = [an.passo(1.0) for _ in range(300)]
print("\n5. Analisador, degrau unitario em C:")
print(f"   medida em 60 s = {y[59]:.3f} (esperado 0)   em 90 s = {y[89]:.3f}"
      f" (esperado 0,632)   em 210 s = {y[209]:.3f} (esperado 0,993)")
