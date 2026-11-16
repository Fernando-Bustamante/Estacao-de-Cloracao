"""
Parte 5: sintonia e teste do override de nivel alto (LC-HH + seletor LS).

Referencia do LC-HH abaixo de 2,7 m, com folga igual ao excesso previsto no
pior caso (valvula de descarga fechada). Cenarios, entre 10 min e 4 h:
  1. rede pressurizada: h_j de 0 para 2,6 m (sem override, o tanque transborda)
  2. falha da valvula de descarga: fica fechada
"""
import numpy as np
import simulador as s
import controladores as c

p = s.Parametros()
Ts, H = 30.0, 3600.0
T_ON, T_OFF = 600.0, 4 * H


def rede(t):
    return {"hj_C": 2.6 if T_ON <= t < T_OFF else 0.0}


class ValvulaTravada:
    """Envolve o controlador e forca a valvula do tanque C fechada."""
    def __init__(self, ctrl):
        self.ctrl = ctrl

    def __call__(self, t, med, u):
        u = self.ctrl(t, med, u)
        if T_ON <= t < T_OFF:
            u[s.VC1] = 0.0
        return u


cenarios = {
    "rede pressurizada (h_j = 2,6 m)": (lambda ct: ct, rede),
    "valvula de descarga fechada": (ValvulaTravada, None),
}

print("tau_c   Kc      Ti      folga prevista   referencia LC-HH")
for tc in (150.0, 300.0):
    so = c.sintonia_override(p, tc, Ts)
    print(f"{tc:5.0f} s {so['Kc']:5.2f}/m  {so['Ti']/60:4.1f} min  {100*so['excesso_previsto']:5.1f} cm"
          f"          {so['h_max']:.2f} m")

for nome, (envolver, pert) in cenarios.items():
    print(f"\nCenario: {nome}")
    for tc in (150.0, 300.0, None):
        sint = c.sintonias_padrao(p, Ts, tau_c_override=tc or 150.0)
        ctrl = envolver(c.Contingencia(p, Ts, sint, override=tc is not None))
        r = s.simular(p, 8 * H, ctrl, perturbacao=pert, Ts=Ts)
        t, h, u1, C = r["t"], r["x"][:, s.HC], r["u"][:, s.PAC], r["x"][:, s.CC]
        rot = f"tau_c = {tc:.0f} s" if tc else "sem override"
        cheio = np.where(h >= p.H - 1e-6)[0]
        extra = f", transborda de {t[cheio[0]]/60:.0f} a {t[cheio[-1]]/60:.0f} min" if len(cheio) else ""
        print(f"  {rot:14s}: h max = {h.max():.3f} m{extra}; u1 min = {u1.min():.3f};"
              f" h em 8 h = {h[-1]:.3f} m; C (com vazao) = {C[(t < T_ON) | (u1 > 0.05)].min():.2f}"
              f" a {C[(t < T_ON) | (u1 > 0.05)].max():.2f} mg/L")

# detalhe do caso adotado no cenario 1: troca sem salto
sint = c.sintonias_padrao(p, Ts, tau_c_override=150.0)
r = s.simular(p, 8 * H, c.Contingencia(p, Ts, sint), perturbacao=rede, Ts=Ts)
u1 = r["u"][::30, s.PAC]
tt = r["t"][::30]
sel = np.where(u1 < u1[0] - 1e-6)[0]
print(f"\nCenario 1, tau_c = 150 s: override assume em {tt[sel[0]]/60:.0f} min e devolve em"
      f" {tt[sel[-1]]/60:.0f} min; maior variacao de u1 entre amostras = {np.abs(np.diff(u1)).max():.3f};"
      f" producao no periodo = {100*u1[(tt>2*H)&(tt<T_OFF)].mean()/u1[0]:.0f}% da nominal")
