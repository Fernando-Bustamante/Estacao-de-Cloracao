"""
Parte 7: cenarios finais, MPC x contingencia lado a lado, no modelo nao linear.

Figuras (uma por cenario, MPC e contingencia na mesma figura):
  fig_cen1_cloro.png     degrau em C_ref e degrau em k_d
  fig_cen2_nivel.png     degrau em h_ref e mudanca de producao
  fig_cen3_falha.png     rede pressurizada (override x restricoes)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simulador as s
import controladores as c
import mpc as M

p = s.Parametros()
Ts, H = 30.0, 3600.0
_, u0 = s.ponto_operacao(p)
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29",
                     "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a"})


def degraus(lista):
    def f(t):
        v = lista[0][1]
        for ti, vi in lista:
            if t >= ti:
                v = vi
        return v
    return f


def par(t_fim, pert=None, **perfis):
    """Roda o cenario com as duas estruturas."""
    rm = s.simular(p, t_fim, M.ControleMPC(p, Ts, **perfis), perturbacao=pert, Ts=Ts)
    rc = s.simular(p, t_fim, c.Contingencia(p, Ts, **perfis), perturbacao=pert, Ts=Ts)
    return rm, rc


def tracar(ax, rm, rc, sx, nome, escala=1.0, offset=0.0, ref=None):
    for r, cor, rot in ((rm, AZUL, "MPC"), (rc, LARANJA, "contingência")):
        y = (r["x"][:, sx] if isinstance(sx, int) else r["u"][:, sx[1]]) * escala - offset
        ax.plot(r["t"] / 60, y, color=cor, lw=1.8, label=rot)
    if ref is not None:
        ax.plot(rm["t"] / 60, ref * escala - offset, color=CINZA, lw=1, ls=(0, (4, 3)), label="referência")
    ax.set_ylabel(nome)


# ------------------------------------------------ cenario 1: cloro
t0 = 600.0
rm1, rc1 = par(2 * H, Cref=degraus([(0, 2.0), (t0, 2.2)]))
kd = lambda t: {"kd_C": (2 if t >= t0 else 1) / 3600}
rm2, rc2 = par(2 * H, pert=kd)

fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
ref1 = np.where(rm1["t"] >= t0, 2.2, 2.0)
tracar(ax[0, 0], rm1, rc1, s.CC, "$C$ [mg/L]", ref=ref1)
tracar(ax[1, 0], rm1, rc1, ("u", s.PBC), "$u_2$ (bomba dosadora)")
tracar(ax[0, 1], rm2, rc2, s.CC, "$C$ [mg/L]", ref=np.full_like(rm2["t"], 2.0))
tracar(ax[1, 1], rm2, rc2, ("u", s.PBC), "$u_2$ (bomba dosadora)")
ax[0, 0].set_title("Degrau de +0,2 mg/L em $C_{ref}$ (R2)", loc="left", fontsize=10)
ax[0, 1].set_title("Degrau em $k_d$ de 1 para 2 h$^{-1}$ (R3)", loc="left", fontsize=10)
for a in ax[0]:
    a.axhspan(1.96, 2.04, color="#e4e3df", alpha=0.6, lw=0) if a is ax[0, 1] else None
ax[0, 1].text(118, 1.965, "faixa de ±2%", ha="right", va="bottom", fontsize=8, color="#5f5e5a")
for a in ax[1]:
    a.set_xlabel("tempo [min]")
    a.set_xlim(0, 120)
ax[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig_cen1_cloro.png", dpi=200)

# ------------------------------------------------ cenario 2: nivel
rm3, rc3 = par(3 * H, href=degraus([(0, 2.0), (t0, 2.25)]))
rm4, rc4 = par(3 * H, prod=degraus([(0, 1.0), (t0, 1.2)]))

fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
ref3 = np.where(rm3["t"] >= t0, 2.25, 2.0)
tracar(ax[0, 0], rm3, rc3, s.HC, "$h$ [m]", ref=ref3)
tracar(ax[1, 0], rm3, rc3, ("u", s.VC1), "$u_3$ (válvula de descarga)")
tracar(ax[0, 1], rm4, rc4, s.HC, "$h$ [m]", ref=np.full_like(rm4["t"], 2.0))
tracar(ax[1, 1], rm4, rc4, s.CC, "$C$ [mg/L]", ref=np.full_like(rm4["t"], 2.0))
ax[0, 0].set_title("Degrau de +0,25 m em $h_{ref}$ (R4)", loc="left", fontsize=10)
ax[0, 1].set_title("Mudança de produção de +20% (R5)", loc="left", fontsize=10)
for a in ax[1]:
    a.set_xlabel("tempo [min]")
    a.set_xlim(0, 180)
ax[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("fig_cen2_nivel.png", dpi=200)

# ------------------------------------------------ cenario 3: falha
T_ON, T_OFF = 600.0, 4 * H
rede = lambda t: {"hj_C": 2.6 if T_ON <= t < T_OFF else 0.0}
rm5, rc5 = par(6 * H, pert=rede)

fig, ax = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
tracar(ax[0], rm5, rc5, s.HC, "$h$ [m]")
ax[0].axhline(2.7, color=CINZA, lw=1, ls=(0, (4, 3)))
ax[0].text(358, 2.705, "limite do R6 (2,7 m)", ha="right", va="bottom", fontsize=8, color="#5f5e5a")
ax[0].axvspan(T_ON / 60, T_OFF / 60, color="#e4e3df", alpha=0.5, lw=0)
ax[0].text(T_ON / 60 + 4, 2.45, "rede pressurizada ($h_j$ = 2,6 m)", ha="left", fontsize=8, color="#5f5e5a")
ax[0].set_ylim(1.55, 2.78)
tracar(ax[1], rm5, rc5, ("u", s.PAC), "$u_1$ (bomba de água)")
tracar(ax[2], rm5, rc5, ("u", s.VC1), "$u_3$ (válvula de descarga)")
ax[0].set_title("Falha: aumento da pressão da rede entre 10 min e 4 h (R6)", loc="left", fontsize=10)
ax[2].set_xlabel("tempo [min]")
ax[2].set_xlim(0, 360)
ax[0].legend(fontsize=8, frameon=False, loc="center right")
fig.tight_layout()
fig.savefig("fig_cen3_falha.png", dpi=200)

# ------------------------------------------------ numeros para a tabela
def acom(t, y, ti, alvo, fx):
    j = t >= ti
    fora = np.where(np.abs(y[j] - alvo) > fx)[0]
    return (t[j][fora[-1]] - ti) / 60 if len(fora) else 0.0


print("                          MPC                     contingencia")
for nome, (rm, rc), sx, alvo, fx in (
        ("R2 ts (min)", (rm1, rc1), s.CC, 2.2, 0.004),
        ("R3 volta (min)", (rm2, rc2), s.CC, 2.0, 0.04),
        ("R4 ts (min)", (rm3, rc3), s.HC, 2.25, 0.005)):
    print(f"  {nome:16s} {acom(rm['t'], rm['x'][:, sx], t0, alvo, fx):6.1f}                  {acom(rc['t'], rc['x'][:, sx], t0, alvo, fx):6.1f}")
print(f"  R3 |dC| (mg/L)   {np.abs(rm2['x'][:, s.CC]-2).max():6.3f}                  {np.abs(rc2['x'][:, s.CC]-2).max():6.3f}")
print(f"  R5 |dC| (mg/L)   {np.abs(rm4['x'][:, s.CC]-2).max():6.3f}                  {np.abs(rc4['x'][:, s.CC]-2).max():6.3f}")
print(f"  R5 |dh| (m)      {np.abs(rm4['x'][:, s.HC]-2).max():6.3f}                  {np.abs(rc4['x'][:, s.HC]-2).max():6.3f}")
print(f"  R6 h max (m)     {rm5['x'][:, s.HC].max():6.3f}                  {rc5['x'][:, s.HC].max():6.3f}")
ev = (rm5["t"] > 2 * H) & (rm5["t"] < T_OFF)
print(f"  R6 producao (%)  {100*rm5['u'][ev, s.PAC].mean()/u0[s.PAC]:6.0f}                  {100*rc5['u'][ev, s.PAC].mean()/u0[s.PAC]:6.0f}")
