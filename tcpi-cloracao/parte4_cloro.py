"""
Parte 4: sintonia e teste da malha de cloro (AIC + estacao de razao FY).

PI pelo metodo IMC na dose, com o analisador aproximado pela regra da metade.
Estudo de tau_c no modelo linear (R2, R3, R7) e testes no modelo nao linear
com a estrutura de contingencia completa (sem o override, que vem na parte 5).
"""
import numpy as np
import control as ct
import warnings
import simulador as s
import controladores as c

warnings.filterwarnings("ignore")
p = s.Parametros()
x0, u0 = s.ponto_operacao(p)
Ts = 30.0
H = 3600.0
sv = ct.tf("s")

# --------------------------------------------- modelo linear da malha
def malha_linear(p, theta_a=None, kd=None, fCB=1.0, h=None):
    """Planta dose -> C (com a razao), perturbacao kd -> C e analisador."""
    theta_a = p.theta if theta_a is None else theta_a
    kd = p.kd if kd is None else kd
    h = p.h_nom if h is None else h
    A_, V_ = s.area(p, h), s.volume(p, h)
    q3, C = p.q3_nom, p.C_nom
    q2 = C * (q3 + kd * V_) / p.CB
    A = np.array([[-q3 / (2 * h * A_), 0.0], [-kd * C * A_ / V_, -(q3 / V_ + kd)]])
    B2 = np.array([[p.K2 / A_], [p.K2 * (fCB * p.CB - C) / V_]])
    u2 = q2 / p.K2
    D = q2 * p.CB / (q3 - q2)
    G = ct.tf(ct.ss(A, B2 * (u2 / D), [[0, 1]], [[0]]))          # dose -> C
    Gd = ct.tf(ct.ss(A, np.array([[0.0], [-C / 3600]]), [[0, 1]], [[0]]))  # kd [1/h] -> C
    Gm = ct.tf(*ct.pade(theta_a + Ts / 2, 6)) / (p.tau_m * sv + 1)
    return ct.minreal(G, verbose=False), ct.minreal(Gd, verbose=False), Gm


def avaliar(sint, **kw):
    G, Gd, Gm = malha_linear(p, **kw)
    Cpi = sint["Kc"] * (1 + 1 / (sint["Ti"] * sv))
    L = Cpi * G * Gm
    m = ct.stability_margins(L)
    t = np.linspace(0, 3 * H, 10801)
    _, y = ct.step_response(ct.feedback(Cpi * G, Gm), t)
    inf = ct.step_info(y, t, SettlingTimeThreshold=0.02)
    _, yd = ct.step_response(Gd * ct.feedback(1, L), t)
    fora = np.where(np.abs(yd) > 0.04)[0]
    volta = t[fora[-1]] if len(fora) else 0.0
    return m[0], m[1], inf["SettlingTime"], inf["Overshoot"], np.abs(yd).max(), volta


print("Estudo de tau_c no modelo linear (nominal)")
print("  R2: ts <= 20 min, Mp <= 10%   R3: |dC| <= 0,2 e volta a +-2% em 30 min   R7: MG >= 2, MF >= 45")
print("  tau_c    Kc      MG    MF     ts      Mp    |dC|max  volta")
for tc in (100, 150, 200, 250, 300):
    si = c.sintonia_cloro_imc(p, tc, Ts)
    mg, mf, ts, mp, dmax, volta = avaliar(si)
    print(f"  {tc:4d} s  {si['Kc']:5.3f}  {mg:5.2f}  {mf:4.1f}  {ts/60:5.1f}min {mp:4.1f}%  {dmax:.3f}   {volta/60:5.1f}min")

TAU_C = 150.0
sc = c.sintonia_cloro_imc(p, TAU_C, Ts)
print(f"\nEscolha: tau_c = {TAU_C:.0f} s")
print(f"  planta dose -> C: K_D = {sc['K_D']:.3f}, tau_C = {sc['tau_C']:.0f} s; regra da metade:"
      f" tau = {sc['tau']:.0f} s, theta = {sc['theta']:.0f} s")
print(f"  Kc = {sc['Kc']:.3f} (mg/L de dose por mg/L de erro),  Ti = {sc['Ti']:.0f} s ({sc['Ti']/60:.1f} min),"
      f"  dose nominal = {sc['D']:.3f} mg/L")

print("\nRobustez no modelo linear (sintonia fixa, R7)")
print("  caso                    MG    MF     ts      Mp    |dC|max  volta")
casos = [("nominal", {}), ("atraso 2 theta", {"theta_a": 2 * p.theta}),
         ("kd = 0,2 1/h", {"kd": 0.2 / 3600}), ("kd = 3 1/h", {"kd": 3 / 3600}),
         ("CB -20%", {"fCB": 0.8})]
for nome, kw in casos:
    mg, mf, ts, mp, dmax, volta = avaliar(sc, **kw)
    print(f"  {nome:20s}  {mg:5.2f}  {mf:4.1f}  {ts/60:5.1f}min {mp:4.1f}%  {dmax:.3f}   {volta/60:5.1f}min")

# ---------------------------------------------- testes nao lineares
sA = c.sintonia_reservatorio(p.KA1 / p.AA, 0.2 * p.K1 / p.KA1 * 2 * u0[s.PAC], 0.10, Ts)
sB = c.sintonia_reservatorio(p.KB1 / p.AB, 0.2 * p.K2 / p.KB1 * 2 * u0[s.PBC], 0.10, Ts)
sn = c.sintonia_nivel_imc(p, 600.0, Ts)


def degraus(lista):
    def f(t):
        v = lista[0][1]
        for ti, vi in lista:
            if t >= ti:
                v = vi
        return v
    return f


def contingencia(perfil_Cref=lambda t: 2.0, perfil_href=lambda t: 2.0,
                 perfil_prod=lambda t: 1.0, escalonar=True):
    resv = c.ControleReservatorios(p, sA, sB, Ts)
    lic = [c.ControleNivelMistura(p, sn, Ts, k) for k in "CD"]
    aic = [c.ControleCloro(p, sc, Ts, k, escalonar) for k in "CD"]

    def f(t, med, u):
        u = u.copy()
        u[s.PAC] = u[s.PAD] = u0[s.PAC] * perfil_prod(t)
        for L, A_ in zip(lic, aic):
            L.ref, A_.ref = perfil_href(t), perfil_Cref(t)
            u = L.atualizar(med, u)
            u = A_.atualizar(med, u)
        return resv.atualizar(med, u)
    return f


def iniciar_em(h):
    """Equilibrio da planta com o tanque C e o D no nivel h e C = 2 mg/L."""
    q = p.__class__(**{**p.__dict__, "h_nom": h})
    return s.ponto_operacao(q)


def acomodacao(t, y, t0, alvo, faixa):
    j = t >= t0
    fora = np.where(np.abs(y[j] - alvo) > faixa)[0]
    return (t[j][fora[-1]] - t0) if len(fora) else 0.0


print("\nModelo nao linear, estrutura de contingencia (LIC + AIC + razao + reservatorios)")
t0 = 600.0

# R2: degraus em C_ref
for C1 in (2.2, 1.8):
    r = s.simular(p, 3 * H, contingencia(perfil_Cref=degraus([(0, 2.0), (t0, C1)])), Ts=Ts)
    C, t = r["x"][:, s.CC], r["t"]
    ts = acomodacao(t, C, t0, C1, 0.02 * abs(C1 - 2.0))
    mp = 100 * max(0.0, np.max((C[t >= t0] - C1) * np.sign(C1 - 2.0))) / abs(C1 - 2.0)
    print(f"  R2  C_ref 2,0 -> {C1}: ts = {ts/60:.1f} min, Mp = {mp:.1f}%,"
          f" u2 de {r['u'][:, s.PBC].min():.3f} a {r['u'][:, s.PBC].max():.3f}")

# R3: degrau em kd de 1 para 2 1/h, em CB e em hj
pert = [("kd 1 -> 2 1/h", lambda t: {"kd_C": (2 if t >= t0 else 1) / 3600}),
        ("kd 1 -> 0,2 1/h", lambda t: {"kd_C": (0.2 if t >= t0 else 1) / 3600}),
        ("CB -20%", lambda t: {"CB": p.CB * (0.8 if t >= t0 else 1)}),
        ("hj 0 -> 0,3 m", lambda t: {"hj_C": 0.3 if t >= t0 else 0.0})]
for nome, fp in pert:
    r = s.simular(p, 6 * H, contingencia(), perturbacao=fp, Ts=Ts)
    C, t = r["x"][:, s.CC], r["t"]
    dC = np.abs(C - 2.0).max()
    volta = acomodacao(t, C, t0, 2.0, 0.04)
    print(f"  R3  {nome:16s}: |dC| max = {dC:.3f} mg/L, volta a +-2% em {volta/60:5.1f} min,"
          f" C final = {C[-1]:.4f}, |dh| max = {np.abs(r['x'][:, s.HC]-2).max():.3f} m")

# R5: mudanca de producao
for sinal in (+1, -1):
    r = s.simular(p, 8 * H, contingencia(perfil_prod=degraus([(0, 1.0), (t0, 1 + 0.2 * sinal)])), Ts=Ts)
    print(f"  R5  producao {sinal*20:+d}%: |dC| max = {np.abs(r['x'][:, s.CC]-2).max():.3f} mg/L,"
          f" |dh| max = {np.abs(r['x'][:, s.HC]-2).max():.3f} m")

# R7: operacao em 1,5 e 2,5 m (degrau em kd), com e sem escalonamento, e atraso 2 theta
for h in (1.5, 2.5):
    for esc in (True, False):
        xh, uh = iniciar_em(h)
        r = s.simular(p, 6 * H, contingencia(perfil_href=lambda t, h=h: h, escalonar=esc), x0=xh, u0=uh,
                      perturbacao=pert[0][1], Ts=Ts)
        C, t = r["x"][:, s.CC], r["t"]
        tail = C[t >= 5 * H]
        print(f"  R7  h = {h} m, escalonamento {'sim' if esc else 'nao'}: |dC| max = {np.abs(C-2).max():.3f},"
              f" volta em {acomodacao(t, C, t0, 2.0, 0.04)/60:5.1f} min,"
              f" oscilacao final = {tail.max()-tail.min():.4f} mg/L")
q = p.__class__(**{**p.__dict__, "theta": 2 * p.theta})
r = s.simular(q, 6 * H, contingencia(), perturbacao=pert[0][1], Ts=Ts)
C, t = r["x"][:, s.CC], r["t"]
print(f"  R7  atraso do analisador 2 theta: |dC| max = {np.abs(C-2).max():.3f},"
      f" volta em {acomodacao(t, C, t0, 2.0, 0.04)/60:.1f} min")
