"""
Tela sinotica da estacao de cloracao (TCPI).

Roda o mesmo modelo nao linear e os mesmos controladores dos scripts do
relatorio (simulador.py, controladores.py e mpc.py) em tempo acelerado, com
desenho do processo, graficos de tendencia e comandos para mudar referencias,
aplicar perturbacoes e falhas e trocar entre MPC e contingencia.

    python sinotico.py
"""
import math
import time
from collections import deque

import numpy as np
import tkinter as tk
from tkinter import ttk, font as tkfont
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import simulador as s
import controladores as c
import mpc as M

# ------------------------------------------------------------------ cores
FUNDO = "#d9d9d5"        # janela
TELA = "#e9e9e5"         # area do sinotico
EQUIP = "#f8f8f5"        # corpo dos equipamentos
CONTORNO = "#55554f"
TEXTO = "#2b2b29"
TEXTO2 = "#6b6b66"
AGUA = "#5b93c9"
HIPO = "#e0973f"
ALARME = "#c62828"
ATENCAO = "#f0c24b"
COR_C, COR_D = "#2a78d6", "#8e5bb5"


def misturar(c1, c2, f):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * f)):02x}" for x, y in zip(a, b))


def br(v, n=2):
    """Numero com virgula decimal."""
    return f"{v:.{n}f}".replace(".", ",")


def relogio(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


# =================================================================== processo
class Processo:
    """Planta + analisadores + controlador, avancando de 1 em 1 s. O
    controlador e chamado a cada 30 s, como em simulador.simular."""

    TS = 30.0

    def __init__(self, p, estrutura="MPC"):
        self.p = p
        x0, self.u0 = s.ponto_operacao(p)
        self.x = list(x0)
        self.u = self.u0.copy()
        self.t = 0
        self.anaC = s.Analisador(p.theta, p.tau_m, 1.0, x0[s.CC])
        self.anaD = s.Analisador(p.theta, p.tau_m, 1.0, x0[s.CD])
        self.sp = {"Cref": p.C_nom, "href": p.h_nom, "prod": 1.0}
        self.pert = {"kd": 1.0, "CB": 100.0, "hj": 0.0}
        self.valvula_travada = False
        self.falha_qp = False          # otimizador do MPC do tanque C nao converge
        self.falha_at = False          # analisador do tanque C sem sinal (NaN)
        self.estrutura = estrutura
        self.ctl = self._criar(estrutura)
        self._n_trocas = 0
        self.hist = deque(maxlen=5000)
        self.limite = {"C": False, "D": False}
        self.alarmes = {}
        self.eventos = []
        self._registrar()

    # ---------------------------------------------------------- controle
    def _criar(self, e):
        perfis = dict(Cref=lambda t: self.sp["Cref"], href=lambda t: self.sp["href"],
                      prod=lambda t: self.sp["prod"])
        if e == "MPC":
            return M.ControleMPC(self.p, self.TS, **perfis)
        return c.Contingencia(self.p, self.TS, **perfis)

    def medidas(self):
        x = self.x
        return {"hA": x[s.HA], "hB": x[s.HB], "hC": x[s.HC], "hD": x[s.HD],
                "CmC": math.nan if self.falha_at else self.anaC.y, "CmD": self.anaD.y}

    def modo(self, k):
        """Quem comanda o tanque k agora: 'MPC' ou 'contingencia'."""
        return self.ctl.modo[k] if self.estrutura == "MPC" else "contingencia"

    def trocar(self, e, aviso=True):
        """Troca a estrutura inicializando os integradores do novo controlador
        com os comandos que estao sendo aplicados agora."""
        if e == self.estrutura:
            return
        novo, med, u = self._criar(e), self.medidas(), self.u
        novo.resv = self.ctl.resv
        if e == "MPC":
            for k, m in novo.mpc.items():
                Cm = med[m.ic]
                m.u_prev = np.array([u[i] for i in m.iu])
                m.x = np.array([med[m.ih], Cm, Cm, Cm, Cm, 0.0, 0.0])
        else:
            for k in "CD":
                novo.inicializar(k, self.t, med, u)
        self.ctl, self.estrutura = novo, e
        self.limite = {"C": False, "D": False}
        self._n_trocas = 0
        nome = "MPC adaptativo" if e == "MPC" else "contingência"
        if aviso:
            self.evento(f"Estrutura trocada para {nome}. Os integradores do novo controlador partiram "
                        "dos comandos atuais; mudanças depois disso são decisões da nova estrutura.")

    def _avisos_controle(self):
        if self.estrutura == "MPC" and self.ctl.eventos:
            # troca automatica: o MPC do tanque com falha ja passou a contingencia;
            # na tela, a estacao inteira passa, para o seletor do topo refletir o estado
            _, k, causa = self.ctl.eventos[0]
            self.trocar("CONT", aviso=False)
            self.evento(f"TROCA AUTOMÁTICA: falha do {causa} do tanque {k}. A estação passou à "
                        "contingência (PI), sem salto nos comandos. Para voltar, selecione "
                        "MPC adaptativo depois de resolver a falha.", "alarme")
        for k, i1 in (("C", s.PAC), ("D", s.PAD)):
            if self.estrutura == "MPC" and self.ctl.modo[k] != "MPC":
                atuando = self.ctl.cont.ovr[k].selecionado
                on = (f"Override do tanque {k}: o nível passou de 2,63 m e o seletor LS "
                      "entregou a bomba de água ao LC-HH.")
                off = f"Override do tanque {k} liberado: a bomba de água voltou a seguir a produção."
            elif self.estrutura == "MPC":
                m = self.ctl.mpc[k]
                h = self.x[s.HC if k == "C" else s.HD]
                atuando = m.u_prev[0] < 0.85 * m.u1_prod and (m.u_prev[2] > 0.95 or h > 2.5)
                on = (f"MPC do tanque {k} está reduzindo a bomba de água (u1 = "
                      f"{100 * m.u_prev[0]:.0f} %) para manter o nível dentro da restrição.")
                off = f"MPC do tanque {k} voltou a seguir a produção pedida."
            else:
                atuando = self.ctl.ovr[k].selecionado
                on = (f"Override do tanque {k}: o nível passou de 2,63 m e o seletor LS "
                      "entregou a bomba de água ao LC-HH.")
                off = f"Override do tanque {k} liberado: a bomba de água voltou a seguir a produção."
            if atuando != self.limite[k]:
                self.evento(on if atuando else off, "atencao" if atuando else "")
            self.limite[k] = atuando

    # ------------------------------------------------------------ planta
    def perturbacoes(self):
        d = s.perturbacoes_nominais(self.p)
        d["kd_C"] = self.pert["kd"] / 3600
        d["CB"] = self.p.CB * self.pert["CB"] / 100
        d["hj_C"] = self.pert["hj"]
        return d

    def passo(self):
        if self.t % int(self.TS) == 0:
            if self.estrutura == "MPC":
                self.ctl.mpc["C"].falha_simulada = self.falha_qp
            u = np.clip(np.asarray(self.ctl(self.t, self.medidas(), self.u.copy()), float), 0.0, 1.0)
            if self.valvula_travada:
                u[s.VC1] = 0.0
            self.u = u
            self._avisos_controle()
        x = s._rk4(self.p, self.x, self.u, self.perturbacoes(), 1.0, True)
        self.x = s._saturar_estados(self.p, x)
        self.anaC.passo(self.x[s.CC])
        self.anaD.passo(self.x[s.CD])
        self.t += 1
        if self.t % 10 == 0:
            self._registrar()

    def vazoes(self):
        p, x, u = self.p, self.x, self.u
        sucA = 1.0 if x[s.HA] > 0 else 0.0
        sucB = 1.0 if x[s.HB] > 0 else 0.0
        q3C = p.Kv * u[s.VC1] * math.sqrt(max(x[s.HC] - self.pert["hj"], 0.0))
        q3D = p.Kv * u[s.VD1] * math.sqrt(max(x[s.HD], 0.0))
        return {"VA1": u[s.VA1], "VB1": u[s.VB1],
                "PAC": u[s.PAC] * sucA, "PAD": u[s.PAD] * sucA,
                "PBC": u[s.PBC] * sucB, "PBD": u[s.PBD] * sucB,
                "VC1": q3C / (2 * p.q3_nom), "VD1": q3D / (2 * p.q3_nom)}

    # -------------------------------------------------------- registro
    def _registrar(self):
        x, u = self.x, self.u
        self.hist.append((self.t, x[s.HC], x[s.HD], x[s.CC], x[s.CD], self.anaC.y, self.anaD.y,
                          u[s.PAC], u[s.PBC], u[s.VC1], self.sp["href"], self.sp["Cref"]))
        checagens = {
            "hC": (x[s.HC] > 2.7, "ALARME: nível do tanque C acima de 2,7 m (requisito R6).",
                   "Nível do tanque C voltou para baixo de 2,7 m."),
            "hD": (x[s.HD] > 2.7, "ALARME: nível do tanque D acima de 2,7 m (requisito R6).",
                   "Nível do tanque D voltou para baixo de 2,7 m."),
            "CC": (not 0.2 <= x[s.CC] <= 5.0, "ALARME: cloro do tanque C fora da faixa da Portaria 888 (0,2 a 5 mg/L).",
                   "Cloro do tanque C voltou para a faixa de 0,2 a 5 mg/L."),
            "CD": (not 0.2 <= x[s.CD] <= 5.0, "ALARME: cloro do tanque D fora da faixa da Portaria 888 (0,2 a 5 mg/L).",
                   "Cloro do tanque D voltou para a faixa de 0,2 a 5 mg/L."),
            "hA": (x[s.HA] < 0.05, "ALARME: reservatório A vazio, as bombas de água ficaram sem sucção.",
                   "Reservatório A voltou a ter água."),
            "hB": (x[s.HB] < 0.05, "ALARME: reservatório B vazio, as bombas dosadoras ficaram sem sucção.",
                   "Reservatório B voltou a ter hipoclorito."),
        }
        for chave, (ativo, on, off) in checagens.items():
            if ativo != self.alarmes.get(chave, False):
                self.evento(on if ativo else off, "alarme" if ativo else "")
            self.alarmes[chave] = ativo

    def evento(self, texto, tipo=""):
        self.eventos.append((self.t, texto, tipo))


# ============================================================ descricoes
DESCR = {
    "A": "Reservatório A (água). Funciona como um integrador puro, com 4 m de diâmetro. "
         "O LIC-A mantém o nível em 2 m com um PI mais feedforward do consumo das bombas.",
    "B": "Reservatório B (solução de hipoclorito de sódio a 10 g/L). Tem 1 m de diâmetro e "
         "também é controlado por PI mais feedforward (LIC-B).",
    "VA1": "VA1: válvula de entrada de água no reservatório A, comandada pelo LIC-A.",
    "VB1": "VB1: válvula de entrada de hipoclorito no reservatório B, comandada pelo LIC-B.",
    "PAC": "PAC: bomba de água do tanque C (u1). Define a produção: q1 = K1·u1, até 10 L/s.",
    "PAD": "PAD: bomba de água do tanque D (u1). Define a produção: q1 = K1·u1, até 10 L/s.",
    "PBC": "PBC: bomba dosadora de hipoclorito do tanque C (u2), até 20 L/h.",
    "PBD": "PBD: bomba dosadora de hipoclorito do tanque D (u2), até 20 L/h.",
    "VC1": "VC1: válvula de descarga do tanque C (u3). Vazão pela lei de Torricelli: q3 = Kv·u3·√(h − hj).",
    "VD1": "VD1: válvula de descarga do tanque D (u3). Vazão pela lei de Torricelli: q3 = Kv·u3·√(h − hj).",
    "C": "Tanque C: tronco de cone com raio de 1 m no fundo, 2 m no topo e 3 m de altura. "
         "A seta preta marca a referência de nível e a linha vermelha o limite de 2,7 m.",
    "D": "Tanque D: igual ao C. Ele não recebe as perturbações do painel, por isso serve de "
         "comparação com o tanque C.",
    "ATC": "AT-C: analisador de cloro. Mede com 60 s de atraso de transporte e 30 s de "
           "constante de tempo, por isso o cloro medido anda atrás do cloro real.",
    "ATD": "AT-D: analisador de cloro do tanque D, igual ao AT-C.",
    "rede": "Rede de distribuição. A pressão da rede (hj) é a coluna d'água que a válvula de "
            "descarga do tanque C precisa vencer.",
}
DESCR_BOLHA = {
    "LIC": "LIC: PI de nível sintonizado por IMC, atuando na válvula de descarga.",
    "LIC-R": "LIC: PI de nível do reservatório (SIMC para integrador) com feedforward do consumo.",
    "FY": "AIC + FY: o AIC (PI) calcula a dose e a estação de razão FY faz u2 = k·u1·D, "
          "para a dosagem acompanhar a vazão de água.",
    "LS": "LS: seletor de mínimo. Recebe a produção pedida e a saída do LC-HH e passa o menor "
          "valor para a bomba de água (override de nível alto).",
    "MPC": "MPC: um único controlador por tanque decide u1, u2 e u3 juntos, prevendo 120 "
           "amostras (1 h) e respeitando as restrições.",
}


# ================================================================== tela
class Sinotico:
    W, H = 780, 572            # sistema de coordenadas base do desenho
    ESC_RES = 26.0             # px por metro nos reservatorios
    ESC_MIS = 48.0             # px por metro nos tanques de mistura

    def __init__(self, raiz):
        self.raiz = raiz
        self.p = s.Parametros()
        self.proc = Processo(self.p, "MPC")
        self.rodando = True
        self.vel = tk.IntVar(value=60)
        self.acum = 0.0
        self.t_real = time.perf_counter()
        self.fase = {}
        self.n_evt = 0
        self.ticks = 0
        self._fontes()
        self._estilo()
        self._montar()
        self.desenhar()
        self._eventos_iniciais()
        self.raiz.after(40, self.ciclo)

    # ------------------------------------------------------- aparencia
    def _fontes(self):
        fam = set(tkfont.families())
        self.fam = next((f for f in ("Segoe UI", "Helvetica Neue", "DejaVu Sans") if f in fam), "TkDefaultFont")
        self.f_txt = (self.fam, 10)
        self.f_neg = (self.fam, 10, "bold")
        self.f_tit = (self.fam, 13, "bold")

    def _estilo(self):
        st = ttk.Style(self.raiz)
        st.theme_use("clam")
        st.configure(".", background=FUNDO, foreground=TEXTO, font=self.f_txt)
        st.configure("TLabelframe", background=FUNDO, bordercolor="#b5b5b0")
        st.configure("TLabelframe.Label", background=FUNDO, foreground=TEXTO, font=self.f_neg)
        st.configure("TButton", background="#ececea", bordercolor="#a9a9a4", padding=(8, 1))
        st.map("TButton", background=[("active", "#f6f6f4")])
        st.configure("TRadiobutton", background=FUNDO)
        st.configure("TCheckbutton", background=FUNDO)
        st.configure("Horizontal.TScale", background="#f1f1ee", troughcolor="#c9c9c4", gripcount=0,
                     sliderlength=16, bordercolor="#a9a9a4", lightcolor="#f1f1ee", darkcolor="#c9c9c4")
        st.configure("Valor.TLabel", font=self.f_neg, width=9, anchor="e")
        st.configure("Muted.TLabel", foreground=TEXTO2)

    def _montar(self):
        r = self.raiz
        r.configure(bg=FUNDO)
        r.columnconfigure(0, weight=3)
        r.columnconfigure(1, weight=2)
        r.rowconfigure(1, weight=1)

        # ---- cabecalho
        cab = ttk.Frame(r, padding=(12, 8, 12, 4))
        cab.grid(row=0, column=0, columnspan=2, sticky="ew")
        cab.columnconfigure(1, weight=1)
        tit = ttk.Frame(cab)
        tit.grid(row=0, column=0, sticky="w")
        ttk.Label(tit, text="Estação de cloração", font=self.f_tit).pack(anchor="w")
        ttk.Label(tit, text="Simulação do modelo não linear · TCPI · UFMG", style="Muted.TLabel").pack(anchor="w")

        est = ttk.Frame(cab)
        est.grid(row=0, column=1, padx=12, sticky="w")
        self.estrutura = tk.StringVar(value="MPC")
        ttk.Label(est, text="Controle:", font=self.f_neg).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(est, text="MPC adaptativo", value="MPC", variable=self.estrutura,
                        command=self._trocar).grid(row=0, column=1, padx=(8, 0))
        ttk.Radiobutton(est, text="Contingência (PI)", value="CONT",
                        variable=self.estrutura, command=self._trocar).grid(row=0, column=2, padx=(8, 0))

        tempo = ttk.Frame(cab)
        tempo.grid(row=0, column=2, sticky="e")
        ttk.Label(tempo, text="vel.", style="Muted.TLabel").pack(side="left", padx=(0, 2))
        for v in (10, 60, 300):
            ttk.Radiobutton(tempo, text=f"{v}×", value=v, variable=self.vel).pack(side="left")
        self.bt_pausa = ttk.Button(tempo, text="Pausar", command=self._pausar, width=9)
        self.bt_pausa.pack(side="left", padx=(10, 4))
        ttk.Button(tempo, text="Reiniciar", command=self._reiniciar, width=9).pack(side="left")
        self.lbl_t = ttk.Label(tempo, text="00:00:00", font=(self.fam, 18, "bold"), width=8, anchor="e")
        self.lbl_t.pack(side="left", padx=(10, 0))

        # ---- sinotico
        esq = ttk.Frame(r, padding=(12, 4, 6, 4))
        esq.grid(row=1, column=0, sticky="nsew")
        esq.rowconfigure(0, weight=1)
        esq.columnconfigure(0, weight=1)
        self.cv = tk.Canvas(esq, bg=TELA, highlightthickness=1, highlightbackground="#b5b5b0")
        self.cv.grid(row=0, column=0, sticky="nsew")
        self.cv.bind("<Configure>", lambda e: self._agendar_redesenho())
        self.txt_ajuda = ""
        self._redesenho = None

        # ---- tendencias
        dir_ = ttk.Frame(r, padding=(6, 4, 12, 4))
        dir_.grid(row=1, column=1, sticky="nsew")
        dir_.rowconfigure(0, weight=1)
        dir_.columnconfigure(0, weight=1)
        self._graficos(dir_)

        # ---- painel de comandos
        pn = ttk.Frame(r, padding=(12, 2, 6, 6))
        pn.grid(row=2, column=0, sticky="nsew")
        for i, wgt in enumerate((5, 5, 4)):
            pn.columnconfigure(i, weight=wgt, uniform="painel")
        self.vars = {}
        f1 = ttk.LabelFrame(pn, text="Referências (tanques C e D)", padding=(8, 2, 8, 4))
        f1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._slider(f1, "Cref", "Cloro", 1.0, 4.0, 0.1, 2.0, lambda v: f"{br(v)} mg/L")
        self._slider(f1, "href", "Nível", 1.5, 2.5, 0.05, 2.0, lambda v: f"{br(v)} m")
        self._slider(f1, "prod", "Produção", 40, 130, 5, 100, lambda v: f"{v:.0f} %")
        f2 = ttk.LabelFrame(pn, text="Perturbações", padding=(8, 2, 8, 4))
        f2.grid(row=0, column=1, sticky="nsew", padx=6)
        self._slider(f2, "kd", "kd no tanque C", 0.2, 3.0, 0.1, 1.0, lambda v: f"{br(v, 1)} 1/h")
        self._slider(f2, "CB", "Hipoclorito em B", 60, 120, 5, 100, lambda v: f"{v:.0f} %")
        self._slider(f2, "hj", "Rede hj (C)", 0.0, 2.8, 0.1, 0.0, lambda v: f"{br(v, 1)} m")
        f3 = ttk.LabelFrame(pn, text="Falhas e cenários", padding=(8, 2, 8, 4))
        f3.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self.trava = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="VC1 travada", variable=self.trava,
                        command=self._travar).pack(anchor="w", pady=(0, 2))
        self.fqp = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="Otimizador C falha", variable=self.fqp,
                        command=self._falhar_qp).pack(anchor="w", pady=(0, 2))
        self.fat = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="Analisador C sem sinal", variable=self.fat,
                        command=self._falhar_at).pack(anchor="w", pady=(0, 2))
        ttk.Button(f3, text="Rede pressurizada",
                   command=lambda: self._aplicar("hj", 2.6)).pack(fill="x", pady=1)
        ttk.Button(f3, text="Cloro +0,2 mg/L",
                   command=lambda: self._aplicar("Cref", self.proc.sp["Cref"] + 0.2)).pack(fill="x", pady=1)
        ttk.Button(f3, text="Tudo nominal", command=self._nominal).pack(fill="x", pady=1)

        # ---- registro de eventos
        lg = ttk.LabelFrame(r, text="O que está acontecendo", padding=(6, 2, 6, 4))
        lg.grid(row=2, column=1, sticky="nsew", padx=(6, 12), pady=(2, 6))
        lg.rowconfigure(0, weight=1)
        lg.columnconfigure(0, weight=1)
        self.log = tk.Text(lg, height=5, wrap="word", bg=EQUIP, fg=TEXTO, relief="flat",
                           font=(self.fam, 9), padx=6, pady=4, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lg, command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set)
        self.log.tag_configure("hora", foreground=TEXTO2)
        self.log.tag_configure("alarme", foreground=ALARME, font=(self.fam, 9, "bold"))
        self.log.tag_configure("atencao", foreground="#8a5a00")
        self._texto_estrutura()

    def _slider(self, pai, chave, rotulo, a, b, passo, inicial, fmt):
        pai.columnconfigure(1, weight=1, minsize=80)
        lin = len(self.vars) % 3
        ttk.Label(pai, text=rotulo).grid(row=lin, column=0, sticky="w", pady=1)
        var = tk.DoubleVar(value=inicial)
        sc = ttk.Scale(pai, from_=a, to=b, variable=var, orient="horizontal", length=120)
        sc.grid(row=lin, column=1, sticky="ew", padx=8)
        val = ttk.Label(pai, style="Valor.TLabel")
        val.grid(row=lin, column=2, sticky="e")

        def mudar(*_):
            v = round(round(var.get() / passo) * passo, 4)
            val.configure(text=fmt(v))
            self._valor(chave, v)

        def soltar(_):
            v = round(round(var.get() / passo) * passo, 4)
            self._registrar_mudanca(chave, v, fmt(v))

        var.trace_add("write", mudar)
        sc.bind("<ButtonRelease-1>", soltar)
        self.vars[chave] = (var, fmt, inicial)
        mudar()

    # --------------------------------------------------------- acoes
    def _valor(self, chave, v):
        sp, pt = self.proc.sp, self.proc.pert
        if chave in ("Cref", "href"):
            sp[chave] = v
        elif chave == "prod":
            sp["prod"] = v / 100
        else:
            pt[chave] = v

    def _registrar_mudanca(self, chave, v, txt):
        nomes = {"Cref": "Referência de cloro", "href": "Referência de nível", "prod": "Produção",
                 "kd": "Consumo de cloro no tanque C (kd)", "CB": "Concentração do hipoclorito em B",
                 "hj": "Pressão da rede do tanque C (hj)"}
        self.proc.evento(f"{nomes[chave]} alterada para {txt}.")

    def _aplicar(self, chave, v):
        var, fmt, _ = self.vars[chave]
        var.set(v)
        self._registrar_mudanca(chave, v, fmt(v))

    def _nominal(self):
        for chave, (var, fmt, ini) in self.vars.items():
            var.set(ini)
        self.trava.set(False)
        self.proc.valvula_travada = False
        self.fqp.set(False)
        self.fat.set(False)
        self.proc.falha_qp = self.proc.falha_at = False
        self.proc.evento("Referências, perturbações e falhas voltaram ao nominal.")

    def _falhar_qp(self):
        on = self.fqp.get()
        self.proc.falha_qp = on
        if not on:
            txt = ("Otimizador do tanque C voltou a funcionar. A estação segue na contingência até o "
                   "operador selecionar MPC adaptativo.")
        elif self.proc.modo("C") == "MPC":
            txt = "Falha: o otimizador (QP) do MPC do tanque C deixou de convergir."
        else:
            txt = ("Falha do otimizador do tanque C ligada. A estação já está na contingência; "
                   "se o MPC for selecionado, a troca automática volta a acontecer.")
        self.proc.evento(txt, "alarme" if on else "")

    def _falhar_at(self):
        on = self.fat.get()
        self.proc.falha_at = on
        self.proc.evento("Falha: o analisador de cloro do tanque C ficou sem sinal. O AIC passa a manual "
                         "com a última dose." if on else
                         "Analisador do tanque C voltou. O AIC retoma o automático sem salto.",
                         "alarme" if on else "")

    def _travar(self):
        self.proc.valvula_travada = self.trava.get()
        self.proc.evento("Falha: válvula VC1 travada fechada. O tanque C não tem mais saída."
                         if self.trava.get() else "Válvula VC1 liberada.", "alarme" if self.trava.get() else "")

    def _trocar(self):
        self.proc.trocar(self.estrutura.get())
        self._texto_estrutura()
        self.desenhar()

    def _texto_estrutura(self):
        self._ajuda("")

    def _ajuda(self, txt):
        self.txt_ajuda = txt
        if "ajuda" in getattr(self, "ids", {}):
            self.cv.itemconfigure(self.ids["ajuda"], text=txt)

    def _pausar(self):
        self.rodando = not self.rodando
        self.bt_pausa.configure(text="Pausar" if self.rodando else "Continuar")
        self.t_real = time.perf_counter()

    def _reiniciar(self):
        self.proc = Processo(self.p, self.estrutura.get())
        for chave, (var, fmt, ini) in self.vars.items():
            var.set(ini)
        self.trava.set(False)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.n_evt = 0
        self._eventos_iniciais()
        self.atualizar_graficos()

    def _eventos_iniciais(self):
        self.proc.evento("Simulação iniciada no ponto de operação nominal: h = 2,00 m e C = 2,00 mg/L "
                         "nos tanques C e D, produção de 6 L/s em cada um.")
        self.proc.evento("As perturbações do painel agem só no tanque C (exceto o hipoclorito, que é comum). "
                         "O tanque D serve de comparação.")

    # ---------------------------------------------------- laco principal
    def ciclo(self):
        agora = time.perf_counter()
        dt_real = min(agora - self.t_real, 0.2)
        self.t_real = agora
        if self.rodando:
            self.acum = min(self.acum + dt_real * self.vel.get(), self.vel.get() * 0.3)
            limite = agora + 0.03
            while self.acum >= 1.0 and time.perf_counter() < limite:
                self.proc.passo()
                self.acum -= 1.0
        if self.estrutura.get() != self.proc.estrutura:     # houve troca automatica
            self.estrutura.set(self.proc.estrutura)
            self._texto_estrutura()
            self.desenhar()
        self.animar(dt_real if self.rodando else 0.0)
        self.atualizar_dinamico()
        self._descarregar_eventos()
        self.ticks += 1
        if self.ticks % 6 == 0:
            self.atualizar_graficos()
        self.raiz.after(40, self.ciclo)

    def _descarregar_eventos(self):
        ev = self.proc.eventos
        if self.n_evt >= len(ev):
            return
        self.log.configure(state="normal")
        for t, txt, tipo in ev[self.n_evt:]:
            self.log.insert("end", relogio(t) + "  ", "hora")
            self.log.insert("end", txt + "\n", tipo)
        self.log.configure(state="disabled")
        self.log.see("end")
        self.n_evt = len(ev)

    # ============================================================ desenho
    def _agendar_redesenho(self):
        if self._redesenho:
            self.raiz.after_cancel(self._redesenho)
        self._redesenho = self.raiz.after(60, self.desenhar)

    def P(self, x, y):
        return self.ox + x * self.k, self.oy + y * self.k

    def pts(self, *xy):
        out = []
        for i in range(0, len(xy), 2):
            out.extend(self.P(xy[i], xy[i + 1]))
        return out

    def F(self, tam, peso="normal"):
        return (self.fam, -max(7, int(round(tam * self.k))), peso)

    def _dica(self, tag, chave, bolha=False):
        def entrar(_):
            txt = DESCR_BOLHA[chave()] if bolha else DESCR[chave]
            self._ajuda(txt)
        self.cv.tag_bind(tag, "<Enter>", entrar)
        self.cv.tag_bind(tag, "<Leave>", lambda _: self._texto_estrutura())

    def desenhar(self):
        cv = self.cv
        cv.delete("all")
        w, h = max(cv.winfo_width(), 200), max(cv.winfo_height(), 150)
        self.k = min(w / self.W, h / self.H)
        self.ox, self.oy = (w - self.W * self.k) / 2, (h - self.H * self.k) / 2
        self.ids = {}
        k = self.k
        lw = max(1, round(1.6 * k))
        cano = max(2, round(4 * k))

        # ---------- tubulacoes
        canos = [
            ([440, 50, 260, 50, 260, 82], AGUA),
            ([690, 50, 520, 50, 520, 91], HIPO),
            ([260, 210, 260, 232, 125, 232, 125, 271], AGUA),
            ([260, 232, 555, 232, 555, 271], AGUA),
            ([125, 293, 125, 318], AGUA), ([555, 293, 555, 318], AGUA),
            ([225, 293, 225, 318], HIPO), ([655, 293, 655, 318], HIPO),
            ([175, 484, 175, 540], "#7f9fb8"), ([605, 484, 605, 540], "#7f9fb8"),
        ]
        for c_, cor in canos:
            cv.create_line(*self.pts(*c_), fill=misturar(cor, TELA, 0.35), width=cano, capstyle="round", joinstyle="round")
        # hipoclorito passa por cima da agua com uma folga visual
        for c_ in ([520, 212, 520, 250, 225, 250, 225, 271], [520, 250, 655, 250, 655, 271]):
            cv.create_line(*self.pts(*c_), fill=TELA, width=cano + max(4, round(5 * k)))
            cv.create_line(*self.pts(*c_), fill=misturar(HIPO, TELA, 0.35), width=cano, capstyle="round", joinstyle="round")
        cv.create_line(*self.pts(555, 238, 555, 262), fill=TELA, width=cano + max(4, round(5 * k)))
        cv.create_line(*self.pts(555, 232, 555, 271), fill=misturar(AGUA, TELA, 0.35), width=cano)
        for x0, y0 in ((125, 318), (555, 318), (225, 318), (655, 318), (260, 82), (520, 91), (175, 540), (605, 540)):
            cv.create_line(*self.pts(x0, y0 - 6, x0, y0 + 2), fill=CONTORNO, width=lw, arrow="last",
                           arrowshape=(6 * k, 7 * k, 3 * k))

        # ---------- particulas de fluxo
        self.caminhos = {
            "VA1": ([440, 50, 260, 50, 260, 78], AGUA),
            "VB1": ([690, 50, 520, 50, 520, 88], HIPO),
            "Aout": ([260, 210, 260, 232], AGUA),
            "PAC": ([260, 232, 125, 232, 125, 316], AGUA),
            "PAD": ([260, 232, 555, 232, 555, 316], AGUA),
            "Bout": ([520, 212, 520, 250], HIPO),
            "PBC": ([520, 250, 225, 250, 225, 316], HIPO),
            "PBD": ([520, 250, 655, 250, 655, 316], HIPO),
            "VC1": ([175, 486, 175, 536], AGUA),
            "VD1": ([605, 486, 605, 536], AGUA),
        }
        self.particulas = {}
        rr = max(1.5, 2.2 * k)
        for nome, (cam, cor) in self.caminhos.items():
            seg, L = [], 0.0
            for i in range(0, len(cam) - 2, 2):
                a, b = cam[i:i + 2], cam[i + 2:i + 4]
                d = math.hypot(b[0] - a[0], b[1] - a[1])
                seg.append((a, b, L, d))
                L += d
            n = max(1, int(L / 24))
            ids = [cv.create_oval(0, 0, 0, 0, fill=misturar(cor, "#000000", 0.25), outline="") for _ in range(n)]
            self.particulas[nome] = (seg, L, ids, rr)
            self.fase.setdefault(nome, 0.0)

        # ---------- fontes e rede
        cv.create_text(*self.P(446, 50), text="Água bruta", anchor="w", font=self.F(12), fill=TEXTO)
        cv.create_text(*self.P(696, 50), text="NaOCl", anchor="w", font=self.F(12), fill=TEXTO)
        for cx, tg in ((175, "rede"), (605, "rede")):
            cv.create_text(*self.P(cx, 548), text="para a rede", font=self.F(10), fill=TEXTO2, tags=tg)
        self.ids["hj"] = cv.create_text(*self.P(175, 562), text="", font=self.F(9), fill=TEXTO2, tags="rede")
        self._dica("rede", "rede")

        # ---------- reservatorios
        self._reservatorio("A", 260, 52, AGUA, "Reservatório A", lado="e")
        self._reservatorio("B", 520, 13, HIPO, "Reservatório B", lado="w")

        # ---------- tanques de mistura
        self._tanque("C", 175)
        self._tanque("D", 605)

        # ---------- bombas e valvulas
        for nome, x in (("PAC", 125), ("PBC", 225), ("PAD", 555), ("PBD", 655)):
            self._bomba(nome, x, 282)
        self._valvula_h("VA1", 350, 50)
        self._valvula_h("VB1", 610, 50)
        self._valvula_v("VC1", 175, 508)
        self._valvula_v("VD1", 605, 508)

        # ---------- percentuais
        for nome, x, anc in (("PAC", 140, "w"), ("PBC", 210, "e"), ("PAD", 570, "w"), ("PBD", 640, "e")):
            self.ids["pct" + nome] = cv.create_text(*self.P(x, 282), text="", anchor=anc, font=self.F(10), fill=TEXTO,
                                                       justify="left" if anc == "w" else "right")
        for nome, x, y, anc in (("VC1", 192, 508, "w"), ("VD1", 622, 508, "w")):
            self.ids["pct" + nome] = cv.create_text(*self.P(x, y), text="", anchor=anc, font=self.F(10), fill=TEXTO)

        # ---------- analisadores
        self._analisador("C", 235.8, 436, 296)
        self._analisador("D", 544.2, 436, 424)

        # ---------- bolhas dos controladores
        mpc = self.estrutura.get() == "MPC"
        bol = [("PAC", 99, 282, "MPC" if mpc else "LS"), ("PBC", 251, 282, "MPC" if mpc else "FY"),
               ("PAD", 529, 282, "MPC" if mpc else "LS"), ("PBD", 681, 282, "MPC" if mpc else "FY"),
               ("VC1", 145, 508, "MPC" if mpc else "LIC"), ("VD1", 575, 508, "MPC" if mpc else "LIC")]
        for alvo, x, y, txt in bol:
            self._bolha(alvo, x, y, txt)

        # ---------- texto de ajuda (descricao da estrutura ou do equipamento sob o mouse)
        self.ids["ajuda"] = cv.create_text(*self.P(390, 300), text=self.txt_ajuda, anchor="n", width=230 * k,
                                           justify="center", font=self.F(10), fill=TEXTO2)
        self.atualizar_dinamico()

    # ------------------------------------------------- equipamentos
    def _reservatorio(self, nome, cx, a, cor, titulo, lado):
        cv, k = self.cv, self.k
        yt, yb = 95, 95 + 4 * self.ESC_RES
        ry = max(3, 0.25 * a)
        tg = nome
        cv.create_rectangle(*self.pts(cx - a, yt, cx + a, yb), fill=EQUIP, outline="", tags=tg)
        cv.create_oval(*self.pts(cx - a, yb - ry, cx + a, yb + ry), fill=EQUIP, outline="", tags=tg)
        self.ids["liq" + nome] = cv.create_rectangle(0, 0, 0, 0, fill=cor, outline="", tags=tg)
        self.ids["liqb" + nome] = cv.create_oval(*self.pts(cx - a, yb - ry, cx + a, yb + ry), fill=cor, outline="", tags=tg)
        self.ids["sup" + nome] = cv.create_oval(0, 0, 0, 0, fill=misturar(cor, "#ffffff", 0.35), outline=CONTORNO,
                                                width=max(1, round(k)), tags=tg)
        lw = max(1, round(1.8 * k))
        cv.create_line(*self.pts(cx - a, yt, cx - a, yb), fill=CONTORNO, width=lw, tags=tg)
        cv.create_line(*self.pts(cx + a, yt, cx + a, yb), fill=CONTORNO, width=lw, tags=tg)
        cv.create_oval(*self.pts(cx - a, yt - ry, cx + a, yt + ry), outline=CONTORNO, width=lw, tags=tg)
        cv.create_arc(*self.pts(cx - a, yb - ry, cx + a, yb + ry), start=180, extent=180, style="arc",
                      outline=CONTORNO, width=lw, tags=tg)
        xt = cx - a - 14 if lado == "e" else cx + a + 14
        cv.create_text(*self.P(xt, 118), text=titulo, anchor=lado, font=self.F(12, "bold"), fill=TEXTO, tags=tg)
        self.ids["txt" + nome] = cv.create_text(*self.P(xt, 138), text="", anchor=lado, font=self.F(12, "bold"),
                                                fill=TEXTO, tags=tg)
        self.ids["geo" + nome] = (cx, a, yt, yb, ry)
        self._dica(tg, nome)

    def _tanque(self, nome, cx):
        cv, k, e = self.cv, self.k, self.ESC_MIS
        yb = 474
        yt = yb - 3 * e
        b, t = self.p.R0 * e, self.p.R1 * e
        rb, rt = 0.2 * b, 0.2 * t
        tg = nome
        cv.create_polygon(*self.pts(cx - t, yt, cx + t, yt, cx + b, yb, cx - b, yb), fill=EQUIP, outline="", tags=tg)
        cv.create_oval(*self.pts(cx - b, yb - rb, cx + b, yb + rb), fill=EQUIP, outline="", tags=tg)
        self.ids["liqb" + nome] = cv.create_oval(*self.pts(cx - b, yb - rb, cx + b, yb + rb), fill=AGUA, outline="", tags=tg)
        self.ids["liq" + nome] = cv.create_polygon(0, 0, 0, 0, 0, 0, fill=AGUA, outline="", tags=tg)
        self.ids["sup" + nome] = cv.create_oval(0, 0, 0, 0, fill=AGUA, outline=CONTORNO, width=max(1, round(k)), tags=tg)
        lw = max(1, round(1.8 * k))
        self.ids["cont" + nome] = [
            cv.create_line(*self.pts(cx - t, yt, cx - b, yb), fill=CONTORNO, width=lw, tags=tg),
            cv.create_line(*self.pts(cx + t, yt, cx + b, yb), fill=CONTORNO, width=lw, tags=tg),
            cv.create_oval(*self.pts(cx - t, yt - rt, cx + t, yt + rt), outline=CONTORNO, width=lw, tags=tg),
            cv.create_arc(*self.pts(cx - b, yb - rb, cx + b, yb + rb), start=180, extent=180, style="arc",
                          outline=CONTORNO, width=lw, tags=tg)]
        # limite de 2,7 m
        hl = 2.7
        rl = s.raio(self.p, hl) * e
        yl = yb - hl * e
        cv.create_line(*self.pts(cx - rl, yl, cx + rl, yl), fill=ALARME, width=max(1, round(k)), dash=(4, 3), tags=tg)
        lado, anc = (cx - rl - 4, "e") if nome == "C" else (cx + rl + 4, "w")
        cv.create_text(*self.P(lado, yl), text="2,7 m", anchor=anc, font=self.F(9), fill=ALARME, tags=tg)
        # titulo e leituras
        cv.create_rectangle(*self.pts(cx - 54, 398, cx + 54, 452), fill=EQUIP, outline="#b5b5b0", tags=tg)
        cv.create_text(*self.P(cx, 408), text=f"Tanque {nome}", font=self.F(10, "bold"), fill=TEXTO2, tags=tg)
        self.ids["h" + nome] = cv.create_text(*self.P(cx, 424), text="", font=self.F(12, "bold"), fill=TEXTO, tags=tg)
        self.ids["c" + nome] = cv.create_text(*self.P(cx, 441), text="", font=self.F(12, "bold"), fill=TEXTO, tags=tg)
        self.ids["sp" + nome] = cv.create_polygon(0, 0, 0, 0, 0, 0, fill=TEXTO, outline="", tags=tg)
        self.ids["geo" + nome] = (cx, yb, e)
        self._dica(tg, nome)

    def _bomba(self, nome, x, y):
        cv, k = self.cv, self.k
        r = 11
        cv.create_oval(*self.pts(x - r, y - r, x + r, y + r), fill=EQUIP, outline=CONTORNO, width=max(1, round(1.8 * k)), tags=nome)
        self.ids["tri" + nome] = cv.create_polygon(*self.pts(x - 7, y - 5, x + 7, y - 5, x, y + 8), fill=CONTORNO,
                                                   outline=CONTORNO, tags=nome)
        self._dica(nome, nome)

    def _valvula_h(self, nome, x, y):
        cv, k = self.cv, self.k
        lw = max(1, round(1.6 * k))
        self.ids["vt" + nome] = [
            cv.create_polygon(*self.pts(x - 12, y - 7, x, y, x - 12, y + 7), outline=CONTORNO, fill=EQUIP, width=lw, tags=nome),
            cv.create_polygon(*self.pts(x + 12, y - 7, x, y, x + 12, y + 7), outline=CONTORNO, fill=EQUIP, width=lw, tags=nome)]
        cv.create_line(*self.pts(x, y, x, y - 12), fill=CONTORNO, width=lw, tags=nome)
        cv.create_arc(*self.pts(x - 9, y - 21, x + 9, y - 3), start=0, extent=180, style="chord",
                      outline=CONTORNO, fill=EQUIP, width=lw, tags=nome)
        self._dica(nome, nome)

    def _valvula_v(self, nome, x, y):
        cv, k = self.cv, self.k
        lw = max(1, round(1.6 * k))
        self.ids["vt" + nome] = [
            cv.create_polygon(*self.pts(x - 7, y - 12, x, y, x + 7, y - 12), outline=CONTORNO, fill=EQUIP, width=lw, tags=nome),
            cv.create_polygon(*self.pts(x - 7, y + 12, x, y, x + 7, y + 12), outline=CONTORNO, fill=EQUIP, width=lw, tags=nome)]
        cv.create_line(*self.pts(x, y, x - 12, y), fill=CONTORNO, width=lw, tags=nome)
        cv.create_text(*self.P(x + 34, y + 14), text=nome, font=self.F(9, "bold"), fill=TEXTO2, tags=nome)
        self._dica(nome, nome)

    def _analisador(self, nome, xw, y, xb):
        cv, k = self.cv, self.k
        tg = "AT" + nome
        x_in = xb if nome == "C" else xb + 60
        cv.create_line(*self.pts(xw, y, x_in, y), fill=CONTORNO, width=max(1, round(k)), dash=(3, 2))
        cv.create_rectangle(*self.pts(xb, y - 18, xb + 60, y + 18), fill=EQUIP, outline=CONTORNO,
                            width=max(1, round(1.4 * k)), tags=tg)
        cv.create_text(*self.P(xb + 30, y - 8), text=f"AT-{nome}", font=self.F(9, "bold"), fill=TEXTO2, tags=tg)
        self.ids["at" + nome] = cv.create_text(*self.P(xb + 30, y + 7), text="", font=self.F(10, "bold"), fill=TEXTO, tags=tg)
        self._dica(tg, tg)

    def _bolha(self, alvo, x, y, txt):
        cv, k = self.cv, self.k
        tg = "b" + alvo
        r = 12.5
        self.ids[tg] = cv.create_oval(*self.pts(x - r, y - r, x + r, y + r), fill=EQUIP, outline=CONTORNO,
                                      width=max(1, round(1.4 * k)), tags=tg)
        self.ids["bt" + alvo] = cv.create_text(*self.P(x, y), text=txt, font=self.F(8.5, "bold"),
                                               fill=TEXTO, tags=tg)
        self._dica(tg, lambda c_=txt: c_, bolha=True)

    # ------------------------------------------------- atualizacao
    def animar(self, dt):
        if not hasattr(self, "particulas"):
            return
        v = self.proc.vazoes()
        frac = dict(v)
        frac["Aout"] = (v["PAC"] + v["PAD"]) / 2
        frac["Bout"] = (v["PBC"] + v["PBD"]) / 2
        cv = self.cv
        for nome, (seg, L, ids, rr) in self.particulas.items():
            f = max(0.0, min(frac[nome], 1.5))
            if f < 0.01:
                for i in ids:
                    cv.itemconfigure(i, state="hidden")
                continue
            self.fase[nome] = (self.fase[nome] + dt * (8 + 70 * f)) % 24
            for j, i in enumerate(ids):
                d = (j * 24 + self.fase[nome]) % L
                for a, b, l0, ls in seg:
                    if d <= l0 + ls:
                        u = (d - l0) / ls if ls else 0
                        x, y = self.P(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
                        break
                cv.coords(i, x - rr, y - rr, x + rr, y + rr)
                cv.itemconfigure(i, state="normal")

    def cor_mistura(self, C):
        return misturar(AGUA, HIPO, min(max(C / 5.0, 0.0), 1.0) * 0.6)

    def atualizar_dinamico(self):
        if not hasattr(self, "ids"):
            return
        cv, pr, p = self.cv, self.proc, self.p
        x, u = pr.x, pr.u
        self.lbl_t.configure(text=relogio(pr.t))
        # reservatorios
        for nome, h, hmax in (("A", x[s.HA], p.HA_max), ("B", x[s.HB], p.HB_max)):
            cx, a, yt, yb, ry = self.ids["geo" + nome]
            yl = yb - min(h, hmax) * self.ESC_RES
            cv.coords(self.ids["liq" + nome], *self.pts(cx - a, yl, cx + a, yb))
            cv.coords(self.ids["sup" + nome], *self.pts(cx - a, yl - ry, cx + a, yl + ry))
            vazio = h < 0.05
            for chave in ("liq", "liqb", "sup"):
                cv.itemconfigure(self.ids[chave + nome], state="hidden" if vazio else "normal")
            cv.itemconfigure(self.ids["txt" + nome], text=f"h = {br(h)} m", fill=ALARME if vazio else TEXTO)
        # tanques de mistura
        for nome, h, C, Cm in (("C", x[s.HC], x[s.CC], pr.anaC.y), ("D", x[s.HD], x[s.CD], pr.anaD.y)):
            cx, yb, e = self.ids["geo" + nome]
            rh = s.raio(p, h) * e
            yl = yb - h * e
            b = p.R0 * e
            cor = self.cor_mistura(C)
            cv.coords(self.ids["liq" + nome], *self.pts(cx - b, yb, cx + b, yb, cx + rh, yl, cx - rh, yl))
            cv.coords(self.ids["sup" + nome], *self.pts(cx - rh, yl - 0.2 * rh, cx + rh, yl + 0.2 * rh))
            for chave, cc in (("liq", cor), ("liqb", cor), ("sup", misturar(cor, "#ffffff", 0.35))):
                cv.itemconfigure(self.ids[chave + nome], fill=cc)
            alto = h > 2.7
            for i in self.ids["cont" + nome]:
                cv.itemconfigure(i, fill=ALARME if alto else CONTORNO) if cv.type(i) == "line" else \
                    cv.itemconfigure(i, outline=ALARME if alto else CONTORNO)
            cv.itemconfigure(self.ids["h" + nome], text=f"h = {br(h)} m", fill=ALARME if alto else TEXTO)
            fora = not 0.2 <= C <= 5.0
            cv.itemconfigure(self.ids["c" + nome], text=f"C = {br(C)} mg/L", fill=ALARME if fora else TEXTO)
            sem_sinal = nome == "C" and pr.falha_at
            cv.itemconfigure(self.ids["at" + nome], text="sem sinal" if sem_sinal else f"{br(Cm)} mg/L",
                             fill=ALARME if sem_sinal else TEXTO)
            hr = pr.sp["href"]
            rr = s.raio(p, hr) * e
            ys = yb - hr * e
            xs = cx - rr - 3
            cv.coords(self.ids["sp" + nome], *self.pts(xs - 9, ys - 5, xs - 9, ys + 5, xs, ys))
        # bombas e valvulas
        v = pr.vazoes()
        for nome, idx in (("PAC", s.PAC), ("PBC", s.PBC), ("PAD", s.PAD), ("PBD", s.PBD)):
            ligada = v[nome] > 0.005
            cv.itemconfigure(self.ids["tri" + nome], fill=CONTORNO if ligada else EQUIP)
            cv.itemconfigure(self.ids["pct" + nome], text=f"{nome}\n{100 * u[idx]:.0f} %")
        for nome, idx in (("VA1", s.VA1), ("VB1", s.VB1), ("VC1", s.VC1), ("VD1", s.VD1)):
            aberta = u[idx] > 0.005
            for i in self.ids["vt" + nome]:
                cv.itemconfigure(i, fill=CONTORNO if aberta else EQUIP)
            if nome in ("VA1", "VB1"):
                continue
            travada = nome == "VC1" and pr.valvula_travada
            cv.itemconfigure(self.ids["pct" + nome], text="travada" if travada else f"{100 * u[idx]:.0f} %",
                             fill=ALARME if travada else TEXTO)
        cv.itemconfigure(self.ids["hj"], text=f"pressão hj = {br(pr.pert['hj'], 1)} m" if pr.pert["hj"] > 0 else "")
        # bolhas em destaque quando o limite de nivel esta atuando
        for tq, alvo in (("C", "bPAC"), ("D", "bPAD")):
            cv.itemconfigure(self.ids[alvo], fill=ATENCAO if pr.limite[tq] else EQUIP)
        # quem comanda cada atuador (um tanque pode ter trocado sozinho para a contingencia)
        for tq, alvos in (("C", (("PAC", "LS"), ("PBC", "FY"), ("VC1", "LIC"))),
                          ("D", (("PAD", "LS"), ("PBD", "FY"), ("VD1", "LIC")))):
            mpc_tq = pr.modo(tq) == "MPC"
            trocou = pr.estrutura == "MPC" and not mpc_tq
            for alvo, ctl in alvos:
                cv.itemconfigure(self.ids["bt" + alvo], text="MPC" if mpc_tq else ctl,
                                 fill=ALARME if trocou else TEXTO)

    # ------------------------------------------------------- graficos
    def _graficos(self, pai):
        matplotlib.rcParams.update({
            "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": "#e2e2de", "grid.linewidth": 0.6,
            "axes.edgecolor": "#8a8984", "axes.labelcolor": TEXTO, "axes.facecolor": EQUIP,
            "xtick.color": TEXTO2, "ytick.color": TEXTO2, "legend.frameon": False})
        self.fig = Figure(figsize=(5.2, 6.0), dpi=100, facecolor=FUNDO, layout="constrained")
        ax = self.fig.subplots(3, 1, sharex=True)
        self.ax = ax
        cinza = "#8a8984"
        self.ln = {
            "hC": ax[0].plot([], [], color=COR_C, lw=1.8, label="tanque C")[0],
            "hD": ax[0].plot([], [], color=COR_D, lw=1.8, label="tanque D")[0],
            "href": ax[0].plot([], [], color=cinza, lw=1, ls=(0, (4, 3)), label="referência")[0],
            "CC": ax[1].plot([], [], color=COR_C, lw=1.8, label="C real")[0],
            "CmC": ax[1].plot([], [], color=COR_C, lw=1, ls=(0, (1, 1.5)), label="C medido (AT)")[0],
            "CD": ax[1].plot([], [], color=COR_D, lw=1.8, label="D real")[0],
            "Cref": ax[1].plot([], [], color=cinza, lw=1, ls=(0, (4, 3)), label="referência")[0],
            "u1": ax[2].plot([], [], color=AGUA, lw=1.6, label="u1 bomba de água")[0],
            "u2": ax[2].plot([], [], color=HIPO, lw=1.6, label="u2 dosadora")[0],
            "u3": ax[2].plot([], [], color=CONTORNO, lw=1.6, label="u3 válvula")[0],
        }
        ax[0].axhline(2.7, color=ALARME, lw=0.9, ls=(0, (2, 2)))
        ax[0].set_ylabel("nível [m]")
        ax[1].set_ylabel("cloro [mg/L]")
        ax[2].set_ylabel("tanque C [%]")
        ax[2].set_xlabel("tempo [min]")
        ax[2].set_ylim(0, 105)
        for a, n in zip(ax, (3, 4, 3)):
            a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=n, fontsize=7.5,
                     handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)
        self.cg = FigureCanvasTkAgg(self.fig, master=pai)
        self.cg.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.cg.get_tk_widget().configure(bg=FUNDO, highlightthickness=0)
        self.atualizar_graficos()

    def atualizar_graficos(self):
        H = np.array(self.proc.hist)
        if len(H) == 0:
            return
        t = H[:, 0] / 60
        janela = 120.0
        t1 = max(janela, t[-1])
        sel = t >= t1 - janela
        t, H = t[sel], H[sel]
        cols = {"hC": 1, "hD": 2, "CC": 3, "CD": 4, "CmC": 5, "u1": 7, "u2": 8, "u3": 9, "href": 10, "Cref": 11}
        for nome, j in cols.items():
            y = H[:, j] * (100 if nome in ("u1", "u2", "u3") else 1)
            self.ln[nome].set_data(t, y)
        self.ax[2].set_xlim(t1 - janela, t1)
        h = H[:, [1, 2, 10]]
        self.ax[0].set_ylim(min(1.4, h.min() - 0.05), max(2.85, h.max() + 0.05))
        Cv = H[:, [3, 4, 5, 11]]
        lo, hi = Cv.min(), Cv.max()
        meio, amp = (lo + hi) / 2, max((hi - lo) / 2 + 0.08, 0.3)
        self.ax[1].set_ylim(meio - 1.3 * amp, meio + amp)
        self.cg.draw_idle()


def main():
    try:  # nitidez em telas com escala no Windows
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    raiz = tk.Tk()
    raiz.title("Estação de cloração · sinótico")
    sw, sh = raiz.winfo_screenwidth(), raiz.winfo_screenheight()
    raiz.geometry(f"{min(1500, sw - 40)}x{min(930, sh - 80)}+10+10")
    raiz.minsize(1100, 680)
    if sh < 950:
        try:
            raiz.state("zoomed")
        except tk.TclError:
            pass
    Sinotico(raiz)
    raiz.mainloop()


if __name__ == "__main__":
    main()
