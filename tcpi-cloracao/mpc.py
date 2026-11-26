"""
MPC adaptativo de um tanque de mistura (Secao 4.5 do relatorio).

Estado do modelo de predicao (7):
    x = [h, C, C_f, z1, z2, d_h, d_C]
    h, C      nivel e concentracao real
    C_f       saida do filtro de primeira ordem do analisador (tau_m)
    z1, z2    atraso de duas amostras (theta = 60 s = 2 Ts):  C_m = z2
    d_h, d_C  perturbacoes integradoras (entram como h_j e como k_d)
Entradas u = [u1, u2, u3]; saidas medidas y = [h, C_m].

O modelo e afim, x[k+1] = A x + B u + c, linearizado a cada amostra em torno
do nivel medido (MPC adaptativo). As variaveis sao absolutas, nao desvios,
para que d_h e d_C sejam interpretaveis como h_j e delta k_d.

Sintonia analitica:
  - pesos pela regra de Bryson, q = 1/(desvio tolerado)^2, com as
    tolerancias dos requisitos R1 a R6;
  - covariancias do filtro de Kalman a partir da incerteza dos medidores e
    da velocidade esperada das perturbacoes.
"""
import math
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import expm, solve_discrete_are
import simulador as s

NX, NU, NY = 7, 3, 2


def pesos_bryson(p, du_max=0.10):
    """Pesos da funcao custo pela regra de Bryson (1/tolerancia^2)."""
    _, u0 = s.ponto_operacao(p)
    return {
        "q_h": 1.0 / 0.10 ** 2,        # R4/R5: 0,10 m
        "q_C": 1.0 / 0.04 ** 2,        # R3: 2% de 2 mg/L
        "r_du": 1.0 / du_max ** 2,     # movimento maximo por amostra
        "w1": 1.0 / u0[s.PAC] ** 2,    # producao pode ir ate zero em emergencia
        "rho": 1.0 / 0.01 ** 2,        # violacao tolerada: 1 cm ou 0,01 mg/L
        "du_max": du_max,
    }


def covariancias_kalman(sigma_h=0.005, sigma_Cm=0.02, dhj=0.01, dkd_h=0.05):
    """R pela incerteza dos medidores; Q das perturbacoes pela variacao
    admitida por amostra (dhj em m, dkd_h em 1/h)."""
    Q = np.diag([1e-8, 1e-8, 1e-10, 1e-10, 1e-10, dhj ** 2, (dkd_h / 3600) ** 2])
    R = np.diag([sigma_h ** 2, sigma_Cm ** 2])
    return Q, R


class MPC:
    def __init__(self, p, Ts=30.0, Np=120, Nc=10, pesos=None, QR=None, tanque="C",
                 adaptativo=True, limites=None):
        self.p, self.Ts, self.Np, self.Nc = p, Ts, Np, Nc
        self.w = pesos or pesos_bryson(p)
        self.Q, self.R = QR or covariancias_kalman()
        self.adaptativo = adaptativo
        self.lim = limites or {"h": (1.5, 2.7), "C": (0.2, 5.0)}
        self.tanque = tanque
        self.iu = (s.PAC, s.PBC, s.VC1) if tanque == "C" else (s.PAD, s.PBD, s.VD1)
        self.ih, self.ic = ("hC", "CmC") if tanque == "C" else ("hD", "CmD")
        _, u0 = s.ponto_operacao(p)
        self.u_prev = np.array([u0[i] for i in self.iu])
        self.x = np.array([p.h_nom, p.C_nom, p.C_nom, p.C_nom, p.C_nom, 0.0, 0.0])
        self.h_ref, self.C_ref, self.u1_prod = p.h_nom, p.C_nom, u0[s.PAC]
        self._h_modelo = None
        self._cache = {}
        self.eps = 0.0
        self.solver_ok = True
        self.falha_simulada = False   # testes e tela: o QP "nao converge"

    # ------------------------------------------------------------ modelo
    def _linearizar(self, h):
        """Matrizes continuas no equilibrio de nivel h, C nominal."""
        p = self.p
        A_, V_ = s.area(p, h), s.volume(p, h)
        C, q3, kd = p.C_nom, p.q3_nom, p.kd
        q2 = C * (q3 + kd * V_) / p.CB
        q1 = q3 - q2
        u_lin = np.array([q1 / p.K1, q2 / p.K2, q3 / (p.Kv * math.sqrt(h))])
        x_lin = np.array([h, C, C])
        Ac = np.array([[-q3 / (2 * h * A_), 0.0, 0.0],
                       [-kd * C * A_ / V_, -(q3 / V_ + kd), 0.0],
                       [0.0, 1 / p.tau_m, -1 / p.tau_m]])
        Bc = np.array([[p.K1 / A_, p.K2 / A_, -p.Kv * math.sqrt(h) / A_],
                       [-p.K1 * C / V_, p.K2 * (p.CB - C) / V_, 0.0],
                       [0.0, 0.0, 0.0]])
        Ec = np.array([[q3 / (2 * h * A_), 0.0],      # d(dh/dt)/d(h_j)
                       [0.0, -C],                     # d(dC/dt)/d(k_d)
                       [0.0, 0.0]])
        return Ac, Bc, Ec, x_lin, u_lin

    def _modelo(self, h):
        h = round(h, 2) if self.adaptativo else self.p.h_nom
        if h in self._cache:
            return self._cache[h]
        Ac, Bc, Ec, x_lin, u_lin = self._linearizar(h)
        M = np.zeros((8, 8))
        M[:3, :3], M[:3, 3:6], M[:3, 6:8] = Ac, Bc, Ec
        Md = expm(M * self.Ts)
        Ad, Bd, Ed = Md[:3, :3], Md[:3, 3:6], Md[:3, 6:8]
        A = np.zeros((NX, NX)); B = np.zeros((NX, NU)); c = np.zeros(NX)
        A[:3, :3], A[:3, 5:7] = Ad, Ed
        A[3, 2] = 1.0; A[4, 3] = 1.0
        A[5, 5] = A[6, 6] = 1.0
        B[:3, :] = Bd
        c[:3] = x_lin - Ad @ x_lin - Bd @ u_lin      # termo afim (equilibrio)
        Cm = np.zeros((NY, NX)); Cm[0, 0] = 1.0; Cm[1, 4] = 1.0
        P = solve_discrete_are(A.T, Cm.T, self.Q, self.R)
        L = P @ Cm.T @ np.linalg.inv(Cm @ P @ Cm.T + self.R)
        # matrizes de predicao: X_j = A^j x + S_j (B u_prev + c) + sum_m S_{j-m} B du_m
        Np, Nc = self.Np, self.Nc
        Apow = [np.eye(NX)]
        for _ in range(Np):
            Apow.append(A @ Apow[-1])
        S = [np.zeros((NX, NX))]
        for j in range(1, Np + 1):
            S.append(S[-1] + Apow[j - 1])
        F = np.vstack([Apow[j] for j in range(1, Np + 1)])             # Np*NX x NX
        G = np.vstack([S[j] for j in range(1, Np + 1)])                # Np*NX x NX (x (B u_prev + c))
        Phi = np.zeros((Np * NX, Nc * NU))
        for j in range(1, Np + 1):
            for m in range(min(j, Nc)):
                Phi[(j - 1) * NX:j * NX, m * NU:(m + 1) * NU] = S[j - m] @ B
        mod = dict(A=A, B=B, c=c, Cm=Cm, L=L, F=F, G=G, Phi=Phi)
        self._cache[h] = mod
        return mod

    # ------------------------------------------------------------- passo
    def passo(self, med):
        if self.falha_simulada:                 # mesmo caminho de uma falha real do OSQP
            self.solver_ok = False
            return self.u_prev.copy()
        p, w, Np, Nc = self.p, self.w, self.Np, self.Nc
        y = np.array([med[self.ih], med[self.ic]])
        mod = self._modelo(med[self.ih])
        A, B, c, Cm, L = mod["A"], mod["B"], mod["c"], mod["Cm"], mod["L"]
        # ---- correcao do filtro de Kalman (x e a predicao x[k|k-1])
        self.x = self.x + L @ (y - Cm @ self.x)
        # ---- predicao livre e matrizes do QP
        F, G, Phi = mod["F"], mod["G"], mod["Phi"]
        x_livre = F @ self.x + G @ (B @ self.u_prev + c)              # Np*NX
        ih = np.arange(Np) * NX + 0
        iC = np.arange(Np) * NX + 1
        iCm = np.arange(Np) * NX + 4
        nz = Nc * NU + 1
        # custo de rastreamento
        H = np.zeros((nz, nz)); f = np.zeros(nz)
        for idx, q, ref in ((ih, w["q_h"], self.h_ref), (iCm, w["q_C"], self.C_ref)):
            Pm = Phi[idx]; e0 = x_livre[idx] - ref
            H[:-1, :-1] += q * Pm.T @ Pm
            f[:-1] += q * Pm.T @ e0
        # movimentos
        H[:-1, :-1] += w["r_du"] * np.eye(Nc * NU)
        # alvo de producao: u1[k+m] = u1_prev + sum_{i<=m} du1_i
        T1 = np.zeros((Nc, Nc * NU))
        for m in range(Nc):
            for i in range(m + 1):
                T1[m, i * NU] = 1.0
        e1 = self.u_prev[0] - self.u1_prod
        H[:-1, :-1] += w["w1"] * T1.T @ T1
        f[:-1] += w["w1"] * T1.T @ (e1 * np.ones(Nc))
        # folga
        H[-1, -1] += w["rho"]
        # ---- restricoes  l <= Aineq z <= u
        Tu = np.zeros((Nc * NU, Nc * NU))                                # u = u_prev + Tu du
        for m in range(Nc):
            for i in range(m + 1):
                Tu[m * NU:(m + 1) * NU, i * NU:(i + 1) * NU] = np.eye(NU)
        rows, lo, up = [], [], []
        # |du| <= du_max
        rows.append(np.hstack([np.eye(Nc * NU), np.zeros((Nc * NU, 1))]))
        lo.append(-w["du_max"] * np.ones(Nc * NU)); up.append(w["du_max"] * np.ones(Nc * NU))
        # 0 <= u <= 1
        rows.append(np.hstack([Tu, np.zeros((Nc * NU, 1))]))
        uprev_rep = np.tile(self.u_prev, Nc)
        lo.append(-uprev_rep); up.append(1.0 - uprev_rep)
        # saidas com folga: lim_min - eps <= y <= lim_max + eps
        for idx, (lmin, lmax) in ((ih, self.lim["h"]), (iC, self.lim["C"])):
            rows.append(np.hstack([Phi[idx], np.ones((Np, 1))]))
            lo.append(lmin - x_livre[idx]); up.append(np.inf * np.ones(Np))
            rows.append(np.hstack([Phi[idx], -np.ones((Np, 1))]))
            lo.append(-np.inf * np.ones(Np)); up.append(lmax - x_livre[idx])
        # eps >= 0
        r = np.zeros((1, nz)); r[0, -1] = 1.0
        rows.append(r); lo.append([0.0]); up.append([np.inf])
        Aineq = sp.csc_matrix(np.vstack(rows))
        l = np.concatenate([np.atleast_1d(v) for v in lo])
        u = np.concatenate([np.atleast_1d(v) for v in up])
        prob = osqp.OSQP()
        prob.setup(sp.csc_matrix(2 * H), 2 * f, Aineq, l, u, verbose=False,
                   eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=False)
        sol = prob.solve()
        ok = sol.info.status_val in (1, 2)
        self.solver_ok = ok
        if ok and sol.x is not None and np.all(np.isfinite(sol.x)):
            du = sol.x[:NU]
            self.eps = sol.x[-1]
        else:
            du = np.zeros(NU)                                          # mantem o comando
        u_new = np.clip(self.u_prev + du, 0.0, 1.0)
        # ---- predicao do filtro para a proxima amostra
        self.x = A @ self.x + B @ u_new + c
        self.u_prev = u_new
        return u_new


class ControleMPC:
    """MPC nos tanques C e D + LIC-A/LIC-B com feedforward nos reservatorios.

    Troca automatica (troca_automatica=True): se o QP de um tanque falha ou
    se o sinal do seu analisador fica invalido, aquele tanque passa a
    estrutura de contingencia no mesmo instante de amostragem, com os
    integradores inicializados pelos comandos que estavam sendo aplicados
    (sem salto). Na falha do analisador o AIC entra em manual com a ultima
    dose. O outro tanque continua no MPC. A volta ao MPC e decisao do
    operador (nova instancia). Cada troca fica registrada em self.eventos
    como (t, tanque, causa)."""

    def __init__(self, p, Ts=30.0, Cref=None, href=None, prod=None,
                 troca_automatica=True, **kw):
        import controladores as c
        self.p = p
        _, self.u0 = s.ponto_operacao(p)
        self.Cref = Cref or (lambda t: p.C_nom)
        self.href = href or (lambda t: p.h_nom)
        self.prod = prod or (lambda t: 1.0)
        st = c.sintonias_padrao(p, Ts)
        self.resv = c.ControleReservatorios(p, st["A"], st["B"], Ts)
        self.mpc = {k: MPC(p, Ts, tanque=k, **kw) for k in "CD"}
        self.troca_automatica = troca_automatica
        self.cont = c.Contingencia(p, Ts, st, self.Cref, self.href, self.prod) \
            if troca_automatica else None
        self.modo = {"C": "MPC", "D": "MPC"}
        self.eventos = []

    def __call__(self, t, med, u):
        import controladores as c
        u = u.copy()
        for k in "CD":
            m = self.mpc[k]
            if self.modo[k] == "MPC":
                causa = None
                if self.troca_automatica and not c.medida_valida(med[m.ic]):
                    causa = "analisador"
                else:
                    m.h_ref, m.C_ref = self.href(t), self.Cref(t)
                    m.u1_prod = self.u0[s.PAC] * self.prod(t)
                    uk = m.passo(med)
                    if self.troca_automatica and not m.solver_ok:
                        causa = "otimizador"
                    else:
                        for i, val in zip(m.iu, uk):
                            u[i] = val
                if causa:
                    # u ainda contem os comandos aplicados ate agora
                    self.cont.inicializar(k, t, med, u)
                    self.modo[k] = "contingencia"
                    self.eventos.append((t, k, causa))
            if self.modo[k] != "MPC":
                u = self.cont.tanque(k, t, med, u)
        return self.resv.atualizar(med, u)
