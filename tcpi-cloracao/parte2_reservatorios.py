"""
Parte 2: sintonia e teste das malhas de nivel dos reservatorios A e B.

Sintonia SIMC para integrador, com tau_c escolhido para que o R8 (desvio de
0,10 m para degrau de +-20% no consumo) seja atendido mesmo sem feedforward.
Cenarios: degrau de +-20% no consumo total com feedforward ideal, com
feedforward 10% errado e sem feedforward.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulador as s
import controladores as c

p = s.Parametros()
x0, u0 = s.ponto_operacao(p)
Ts = 30.0
DH_MAX = 0.10                     # R8
DEGRAU = 0.20                     # +-20% no consumo total
T0 = 600.0                        # instante do degrau

# ------------------------------------------------------------- sintonia
kA = p.KA1 / p.AA
kB = p.KB1 / p.AB
gA, gB = p.K1 / p.KA1, p.K2 / p.KB1
dA = DEGRAU * gA * (u0[s.PAC] + u0[s.PAD])      # degrau de consumo em unidades de vA1
dB = DEGRAU * gB * (u0[s.PBC] + u0[s.PBD])      # idem em unidades de vB1
sA = c.sintonia_reservatorio(kA, dA, DH_MAX, Ts)
sB = c.sintonia_reservatorio(kB, dB, DH_MAX, Ts)

print("Sintonia (SIMC para integrador, tau_c pelo R8 sem feedforward)")
for nome, k, d, si in [("A", kA, dA, sA), ("B", kB, dB, sB)]:
    print(f"  {nome}: k' = {k:.4e} m/s   delta_d = {d:.4f}   t_deriva = {si['t_deriva']:.0f} s"
          f" ({si['t_deriva']/3600:.2f} h)")
    print(f"     tau_c = {si['tau_c']:.0f} s   Kc = {si['Kc']:.3f} /m   Ti = {si['Ti']:.0f} s"
          f" ({si['Ti']/3600:.2f} h)   pico previsto sem FF = {100*si['pico_sem_ff']:.2f} cm")

# ------------------------------------------------------------ cenarios
def controlador_teste(fator_ff, sinal):
    ctrl = c.ControleReservatorios(p, sA, sB, Ts, fator_ff=fator_ff)

    def f(t, med, u):
        u = u0.copy()
        if t >= T0:
            for i in (s.PAC, s.PBC, s.PAD, s.PBD):
                u[i] = u0[i] * (1 + sinal * DEGRAU)
        return ctrl.atualizar(med, u)
    return f


cenarios = [("feedforward ideal", 1.0, "-"),
            ("feedforward com erro de 10%", 0.9, "--"),
            ("sem feedforward", 0.0, ":")]
cores = ["#2a78d6", "#eb6834", "#1baf7a"]
T_FIM = 12 * 24 * 3600            # 12 dias, para mostrar o reservatorio B

res = {}
print("\nResultados (degrau no consumo em t = 10 min)")
print("  cenario                          sinal  |dhA| max   vA1 min-max    |dhB| max   vB1 min-max    dhA final  dhB final")
for nome, f_ff, _ in cenarios:
    for sinal in (+1, -1):
        r = s.simular(p, T_FIM, controlador_teste(f_ff, sinal), Ts=Ts)
        dhA = r["x"][:, s.HA] - p.hA_nom
        dhB = r["x"][:, s.HB] - p.hB_nom
        vA, vB = r["u"][:, s.VA1], r["u"][:, s.VB1]
        res[(nome, sinal)] = r
        print(f"  {nome:30s}  {sinal:+d}   {100*abs(dhA).max():6.2f} cm  {vA.min():.3f}-{vA.max():.3f}"
              f"   {100*abs(dhB).max():6.2f} cm  {vB.min():.3f}-{vB.max():.3f}"
              f"   {100*dhA[-1]:+.3f} cm  {100*dhB[-1]:+.3f} cm")

# --------------------------------------------------------------- figura
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29",
                     "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a"})
fig, ax = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col")
for (nome, _, ls), cor in zip(cenarios, cores):
    r = res[(nome, +1)]
    tm, th = r["t"] / 60, r["t"] / 3600
    ax[0, 0].plot(tm, 100 * (r["x"][:, s.HA] - p.hA_nom), color=cor, ls=ls, lw=2, label=nome)
    ax[1, 0].plot(tm, r["u"][:, s.VA1], color=cor, ls=ls, lw=2)
    ax[0, 1].plot(th, 100 * (r["x"][:, s.HB] - p.hB_nom), color=cor, ls=ls, lw=2, label=nome)
    ax[1, 1].plot(th, r["u"][:, s.VB1], color=cor, ls=ls, lw=2)

for a in (ax[0, 0], ax[0, 1]):
    a.axhline(-100 * DH_MAX, color="#8a8984", lw=1, ls=(0, (4, 3)))
    a.text(0.99, -9.6, "limite do R8 (-10 cm)", transform=a.get_yaxis_transform(),
           ha="right", va="bottom", fontsize=8, color="#5f5e5a")
    a.set_ylim(-11, 1.5)
ax[0, 0].set_xlim(0, 240)
ax[0, 1].set_xlim(0, 288)
ax[0, 0].set_title("Reservatório A (água bruta)", fontsize=10, loc="left")
ax[0, 1].set_title("Reservatório B (solução de NaOCl)", fontsize=10, loc="left")
ax[0, 0].set_ylabel("$h_A - \\bar h_A$ [cm]")
ax[0, 1].set_ylabel("$h_B - \\bar h_B$ [cm]")
ax[1, 0].set_ylabel("$v_{A1}$ [-]")
ax[1, 1].set_ylabel("$v_{B1}$ [-]")
ax[1, 0].set_xlabel("tempo [min]")
ax[1, 1].set_xlabel("tempo [h]")
ax[0, 0].legend(loc="center right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("fig_reservatorios.png", dpi=200)
print("\nFigura salva em fig_reservatorios.png")
