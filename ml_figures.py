#!/usr/bin/env python3
"""
Модель Морриса–Лекара: пороги при токовом и кондуктансном импульсных входах.
Воспроизводит рис. 1–3 статьи для журнала «Известия вузов. Радиофизика».
Запуск:  python ml_figures.py          (полный расчёт, около 1–2 мин на обычном ПК)
         python ml_figures.py --replot (только перерисовка из output/cache.npz, несколько секунд)
Выход:   output/fig1.pdf|png, fig2.pdf|png, fig3.pdf|png, thresholds_ML.csv, cache.npz
Зависимости: numpy, scipy, matplotlib, pandas.
Интегратор: РК4 с фиксированным шагом dt = 0.1 мс (векторизован по набору амплитуд).
Единицы: V — мВ, t — мс, I — мкА/см^2, G — мСм/см^2, C — мкФ/см^2.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy.optimize import brentq, curve_fit, minimize_scalar

OUT = "output"
os.makedirs(OUT, exist_ok=True)
MM = 1 / 25.4                 # мм -> дюймы; ширина рисунка 80 или 160 мм
VTH = 20.0                    # критерий спайка: V_max > +20 мВ
VREV = 0.0                    # реверсивный потенциал возбуждающего входа, мВ
DT = 0.1

# ----------------------------- параметры модели -----------------------------
BASE = dict(C=20., gL=2., VK=-84., VL=-60., VCa=120., V1=-1.2, V2=18.)
SET_I = dict(BASE, gCa=4.0, gK=8., V3=12., V4=17.4, phi=0.067)   # класс I  (SNIC)
SET_II = dict(BASE, gCa=4.4, gK=8., V3=2., V4=30., phi=0.04)     # класс II (Хопф)
SETS = {"I": SET_I, "II": SET_II}

def minf(V, p): return 0.5 * (1 + np.tanh((V - p["V1"]) / p["V2"]))
def winf(V, p): return 0.5 * (1 + np.tanh((V - p["V3"]) / p["V4"]))
def tauw(V, p): return 1 / np.cosh((V - p["V3"]) / (2 * p["V4"]))

def Iion(V, w, p):
    return (-p["gCa"] * minf(V, p) * (V - p["VCa"]) - p["gK"] * w * (V - p["VK"])
            - p["gL"] * (V - p["VL"]))

def rest(I, p):
    """Равновесия при токе I (корни V-нуль-клины при w = w_inf(V))."""
    f = lambda V: Iion(V, winf(V, p), p) + I
    Vs = np.linspace(-80, 60, 5601); fs = f(Vs)
    return [brentq(f, Vs[i], Vs[i + 1]) for i in range(len(Vs) - 1) if fs[i] * fs[i + 1] < 0]

def jac(V, I, p):
    w = winf(V, p); h = 1e-5
    dV = lambda V_, w_: (Iion(V_, w_, p) + I) / p["C"]
    dw = lambda V_, w_: p["phi"] * (winf(V_, p) - w_) / tauw(V_, p)
    return np.array([
        [(dV(V + h, w) - dV(V - h, w)) / (2 * h), (dV(V, w + h) - dV(V, w - h)) / (2 * h)],
        [(dw(V + h, w) - dw(V - h, w)) / (2 * h), (dw(V, w + h) - dw(V, w - h)) / (2 * h)]])

# ----------------------------- интегратор -----------------------------------
def sim(p, mode, amp, tau, Vrev=VREV, Ib=0., Tpost=150., dt=DT, retQ=False):
    """Импульс длительностью tau на фоне тока Ib. mode: 'I' (ток A) или 'G' (проводимость G).
    Для 'G' ток стимула = G (Vrev - V). Возвращает максимум V (и перенесённый заряд)."""
    amp = np.atleast_1d(np.asarray(amp, float)); N = amp.size
    tau = np.broadcast_to(np.asarray(tau, float), (N,))
    Vrev = np.broadcast_to(np.asarray(Vrev, float), (N,))
    Ib = np.broadcast_to(np.asarray(Ib, float), (N,))
    cache = {b: rest(b, p)[0] for b in np.unique(Ib)}
    V = np.array([cache[b] for b in Ib]); w = winf(V, p)
    Vmax = V.copy(); Q = np.zeros(N)
    ntau = np.round(tau / dt).astype(int); nmax = int((tau.max() + Tpost) / dt)
    for k in range(nmax):
        on = (k < ntau).astype(float)
        def f(V, w):
            Is = amp * on if mode == "I" else amp * on * (Vrev - V)
            return (Iion(V, w, p) + Ib + Is) / p["C"], p["phi"] * (winf(V, p) - w) / tauw(V, p), Is
        k1 = f(V, w); k2 = f(V + .5 * dt * k1[0], w + .5 * dt * k1[1])
        k3 = f(V + .5 * dt * k2[0], w + .5 * dt * k2[1]); k4 = f(V + dt * k3[0], w + dt * k3[1])
        Q += dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
        V = V + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Vmax = np.maximum(Vmax, V)
    return (Vmax, Q) if retQ else Vmax

def threshold(p, mode, taus, Vrev=VREV, ng=100, nb=22, gr=None, dt=DT, Tpost=150., full=False):
    """Нижний порог (скан по сетке + бисекция) и верхняя граница спайкового окна."""
    taus = np.atleast_1d(np.asarray(taus, float)); nt = taus.size
    gr = gr or ((5, 1500) if mode == "I" else (0.05, 80))
    g = np.geomspace(*gr, ng)
    Vr = np.broadcast_to(np.asarray(Vrev, float), (nt,))
    vm = sim(p, mode, np.repeat(g[None, :], nt, 0).ravel(), np.repeat(taus[:, None], ng, 1).ravel(),
             np.repeat(Vr[:, None], ng, 1).ravel(), dt=dt, Tpost=Tpost).reshape(nt, ng)
    sp = vm > VTH; valid = sp.any(1); first = np.where(valid, sp.argmax(1), 0)
    lo = np.where(first > 0, g[np.maximum(first - 1, 0)], 0.); hi = np.where(valid, g[first], 1.)
    for _ in range(nb):
        mid = (lo + hi) / 2; s = sim(p, mode, mid, taus, Vr, dt=dt, Tpost=Tpost) > VTH
        hi = np.where(s, mid, hi); lo = np.where(s, lo, mid)
    last = np.where(valid, ng - 1 - sp[:, ::-1].argmax(1), 0)
    blk = np.where(valid, np.where(last < ng - 1, g[np.minimum(last + 1, ng - 1)], np.inf), np.nan)
    if not full:
        return np.where(valid, hi, np.nan), blk
    # верхняя граница ОСНОВНОГО окна: первый узел без спайка после первого спайкового узла,
    # уточнённая бисекцией между последним спайковым и первым неспайковым узлом.
    # blk (выше) — край ПОСЛЕДНЕГО спайкового интервала на сетке; он отличается от основного окна,
    # если область спайка состоит из двух интервалов (класс I, tau = 60 и 100 мс).
    after = (~sp) & (np.arange(ng)[None, :] > first[:, None]); has = valid & after.any(1)
    e = np.where(has, after.argmax(1), 1)
    lo2 = np.where(has, g[np.maximum(e - 1, 0)], 1.); hi2 = np.where(has, g[e], 2.)
    for _ in range(nb):
        mid = (lo2 + hi2) / 2; s_ = sim(p, mode, mid, taus, Vr, dt=dt, Tpost=Tpost) > VTH
        lo2 = np.where(s_, mid, lo2); hi2 = np.where(s_, hi2, mid)
    ubm = np.where(has, (lo2 + hi2) / 2, np.where(valid, np.inf, np.nan))
    return np.where(valid, hi, np.nan), blk, ubm

def traj(p, mode, amp, tau, T=120., dt=0.05):
    V = rest(0., p)[0]; w = winf(V, p); Vs = [V]; ws = [w]; nt = int(tau / dt)
    for k in range(int(T / dt)):
        on = 1. if k < nt else 0.
        def f(V, w):
            Is = amp * on if mode == "I" else amp * on * (VREV - V)
            return (Iion(V, w, p) + Is) / p["C"], p["phi"] * (winf(V, p) - w) / tauw(V, p)
        k1 = f(V, w); k2 = f(V + .5 * dt * k1[0], w + .5 * dt * k1[1])
        k3 = f(V + .5 * dt * k2[0], w + .5 * dt * k2[1]); k4 = f(V + dt * k3[0], w + dt * k3[1])
        V += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); w += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        Vs.append(V); ws.append(w)
    return np.array(Vs), np.array(ws)

def sweep(p, Ibs, V, w, dt=0.2, settle=800.):
    """Медленное изменение тока; возвращает размах колебаний (гистерезис)."""
    amps = []; n = int(settle / dt)
    for Ib in Ibs:
        vmx, vmn = -1e9, 1e9
        for k in range(n):
            f = lambda V, w: ((Iion(V, w, p) + Ib) / p["C"], p["phi"] * (winf(V, p) - w) / tauw(V, p))
            k1 = f(V, w); k2 = f(V + .5 * dt * k1[0], w + .5 * dt * k1[1])
            k3 = f(V + .5 * dt * k2[0], w + .5 * dt * k2[1]); k4 = f(V + dt * k3[0], w + dt * k3[1])
            V = V + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); w = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            if k > 0.6 * n: vmx, vmn = max(vmx, V), min(vmn, V)
        amps.append(vmx - vmn)
    return np.array(amps)

# ----------------------------- оформление по требованиям журнала ------------
# ширина 160 мм, векторный PDF (шрифты TrueType/Type 42 встроены) + PNG 600 dpi, без сетки;
# оси подписаны символом величины и единицей («V, мВ»); при печати в 160 мм все надписи,
# включая индексы и степени (0.7 от кегля подписи оси), не мельче 6.5 pt.
FS = 9.5                      # подписи осей, обозначения панелей; индексы: 0.7 * 9.5 = 6.65 pt
FS_S = 8.0                    # деления осей и легенды (без индексов)
DEC_COMMA = True              # десятичная запятая на осях (русскоязычный журнал); False — точка
HYST = (88.3, 94.4)           # окно гистерезиса класса II, мкА/см^2 (значения из главы 3; сверяются с рис. 3в)
# цвета подобраны так, чтобы пары различались и в градациях серого (светлота L* ~ 35 и 65; 27 и 47),
# дополнительно классы различаются маркерами, направления развёртки — заливкой маркеров
COL = {"I": "#00602b", "II": "#b58be0"}      # класс I — тёмно-зелёный, класс II — светло-фиолетовый
MRK = {"I": "o", "II": "^"}
C_SUB, C_SUP = "#0b3c8c", "#d62728"          # тёмно-синий (подпороговый / I растёт), красный (спайк / I убывает)
plt.rcParams.update({"font.size": FS_S, "axes.labelsize": FS, "axes.titlesize": FS, "legend.fontsize": FS_S,
                     "xtick.labelsize": FS_S, "ytick.labelsize": FS_S,
                     "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.grid": False, "legend.frameon": False,
                     "legend.handlelength": 2.4, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

def _num(x, _=None, dec=None):
    s = f"{x:.10g}" if dec is None else f"{x:.{dec}f}"
    return (s.replace(".", ",") if DEC_COMMA else s).replace("-", "\u2212")
FMT = FuncFormatter(_num)

def fmt_axes(a, xt=None, yt=None, ydec=None):
    """Числа на осях без показателей степени; xt, yt — явные деления для логарифмических осей,
    ydec — фиксированное число десятичных знаков на оси ординат."""
    for axis, ticks in ((a.xaxis, xt), (a.yaxis, yt)):
        if ticks is not None:
            axis.set_ticks(ticks); axis.set_minor_formatter(NullFormatter())
        axis.set_major_formatter(FMT)
    if ydec is not None:
        a.yaxis.set_major_formatter(FuncFormatter(lambda x, _: _num(x, dec=ydec)))

def panel(a, text):
    a.set_title(text, loc="left", pad=3)

def save(fig, name):
    fig.tight_layout(pad=0.3, w_pad=0.9)
    fig.savefig(f"{OUT}/{name}.pdf", dpi=600); fig.savefig(f"{OUT}/{name}.png", dpi=600); plt.close(fig)

TAUS = np.array([0.7, 1, 1.5, 2, 3, 5, 8, 12, 20, 35, 60, 100.])
VREVS = np.arange(-5, 81, 5.); TAU0 = 5.
IB_DN = np.arange(98, 84.9, -0.5); IB_UP = np.arange(84, 98.01, 0.5)
fit_fn = lambda Vr, Qc, Vs: Qc / (TAU0 * (Vr - Vs))

# ----------------------------- расчёт ---------------------------------------
def compute():
    D = {}
    # --- бифуркации
    D["IH"] = brentq(lambda I: np.trace(jac(rest(I, SET_II)[0], I, SET_II)), 60, 120)
    D["VH"] = rest(D["IH"], SET_II)[0]
    # точка складки: максимум I(V) = -Iion(V, w_inf(V)) на кривой равновесий
    r = minimize_scalar(lambda V: Iion(V, winf(V, SET_I), SET_I), bounds=(-45, -15),
                        method="bounded", options={"xatol": 1e-10})
    D["ISN"], D["VSN"] = -r.fun, r.x
    # статический седло-узел под кондуктансной нагрузкой: максимум G(V) = -Iion / (Vrev - V)
    D["GSN"] = -minimize_scalar(lambda V: Iion(V, winf(V, SET_I), SET_I) / (VREV - V), bounds=(-45, -15),
                                method="bounded", options={"xatol": 1e-10}).fun
    # --- пороги
    for c in SETS:
        D[f"V0_{c}"] = rest(0., SETS[c])[0]
        D[f"A_{c}"] = threshold(SETS[c], "I", TAUS)[0]
        D[f"G_{c}"], D[f"B_{c}"], D[f"M_{c}"] = threshold(SETS[c], "G", TAUS, full=True)
    # --- рис. 1: траектории (0.9 и 1.1 порога, tau = 5 мс) и кривые равновесий I(V) с устойчивостью
    Vp = np.linspace(-70, 25, 1901); D["eqV"] = Vp
    for c, p in SETS.items():
        Ath = D[f"A_{c}"][TAUS == 5][0]
        for k, fr in enumerate((0.9, 1.1)):
            D[f"trV_{c}_{k}"], D[f"trw_{c}_{k}"] = traj(p, "I", fr * Ath, 5.)
        Ip = -Iion(Vp, winf(Vp, p), p); D[f"eqI_{c}"] = Ip
        st = []
        for V, I in zip(Vp, Ip):
            J = jac(V, I, p); st.append(np.trace(J) < 0 and np.linalg.det(J) > 0)
        D[f"eqS_{c}"] = np.array(st)
    # --- рис. 2в: карта V_max(tau, G), класс I (те же dt и Tpost, что и для порогов)
    tm = np.geomspace(0.7, 100, 120); gm = np.geomspace(0.3, 80, 240)
    D["tm"], D["gm"] = tm, gm
    D["vmap"] = sim(SET_I, "G", np.repeat(gm[None, :], tm.size, 0).ravel(),
                    np.repeat(tm[:, None], gm.size, 1).ravel()).reshape(tm.size, gm.size)
    # --- рис. 3: G_th(V_rev), подгонка, гистерезис по фоновому току
    for c in SETS:
        D[f"thG_{c}"] = threshold(SETS[c], "G", np.full(VREVS.size, TAU0), Vrev=VREVS, gr=(0.05, 300), ng=120)[0]
        D[f"fit_{c}"], _ = curve_fit(fit_fn, VREVS, D[f"thG_{c}"], p0=[1000, -30])
    D["a_dn"] = sweep(SET_II, IB_DN, -20., 0.2)
    V0 = rest(84., SET_II)[0]; D["a_up"] = sweep(SET_II, IB_UP, V0, winf(V0, SET_II))
    # --- проверка переданного заряда на пороге (tau = 5 мс)
    for c, p in SETS.items():
        th_G = D[f"G_{c}"][TAUS == 5][0]; th_A = D[f"A_{c}"][TAUS == 5][0]
        _, QG = sim(p, "G", th_G * 1.002, 5., retQ=True)
        D[f"Qratio_{c}"] = QG[0] / (th_A * 5.)
    return D

def report(D):
    pd.DataFrame([dict(класс=c, tau_ms=t, A_th=D[f"A_{c}"][i], G_th=D[f"G_{c}"][i], G_up_main=D[f"M_{c}"][i], G_up_last=D[f"B_{c}"][i])
                  for c in SETS for i, t in enumerate(TAUS)]).to_csv(f"{OUT}/thresholds_ML.csv", index=False)
    print(f"I_SN(класс I) = {D['ISN']:.2f}, I_H(класс II) = {D['IH']:.2f}")
    for c in SETS:
        print(f"класс {c}: Q = {D[f'fit_{c}'][0]:.0f} нКл/см2, V* = {D[f'fit_{c}'][1]:.1f} мВ")
    for c in SETS:
        print(f"класс {c}: Q_G/Q_I на пороге = {float(D[f'Qratio_{c}']):.3f}")
    # --- дополнительные числа для сверки с текстом главы 3
    print(f"V_SN = {D['VSN']:.1f} мВ, V_H = {D['VH']:.1f} мВ, G_SN = {D['GSN']:.3f} мСм/см2")
    for c in SETS:
        A, G = D[f"A_{c}"], D[f"G_{c}"]; m = ~np.isnan(G); V0 = float(D[f"V0_{c}"])
        r = G[m] * (VREV - V0) / A[m]
        rms = np.sqrt(np.mean((fit_fn(VREVS, *D[f"fit_{c}"]) / D[f"thG_{c}"] - 1) ** 2))
        print(f"класс {c}: V0 = {V0:.1f} мВ; A_th(100 мс) = {A[-1]:.1f}; A_th*tau(1 мс) = {A[1]:.0f}; "
              f"G_th(max tau) = {G[m][-1]:.2f} при tau = {TAUS[m][-1]:g} мс; r = {r[0]:.2f}…{r[-1]:.2f}; "
              f"СКО подгонки = {100 * rms:.1f}%")
    up, dn = D["a_up"], D["a_dn"]; iu = np.argmax(up > 40); idn = np.argmax(dn < 40)
    print(f"гистерезис (шаг 0.5): возникновение колебаний между {IB_UP[iu - 1]:.1f} и {IB_UP[iu]:.1f}, "
          f"срыв между {IB_DN[idn]:.1f} и {IB_DN[idn - 1]:.1f} мкА/см2; размах = {dn[IB_DN == 90][0]:.1f} мВ (I = 90)")
    if not (IB_DN[idn] <= HYST[0] <= IB_DN[idn - 1] and IB_UP[iu - 1] <= HYST[1] <= IB_UP[iu]):
        print("ВНИМАНИЕ: окно HYST не согласуется с развёрткой рис. 3в")

# ----------------------------- рисунки --------------------------------------
def plot(D):
    ISN, IH, GSN = float(D["ISN"]), float(D["IH"]), float(D["GSN"])
    W, H = 160 * MM, 62 * MM

    # --- Рис. 1
    fig, ax = plt.subplots(1, 3, figsize=(W, H))
    Vg = np.linspace(-80, 60, 600)
    for a, (c, lab) in zip(ax[:2], [("I", "а"), ("II", "б")]):
        p = SETS[c]
        wV = (-p["gCa"] * minf(Vg, p) * (Vg - p["VCa"]) - p["gL"] * (Vg - p["VL"])) / (p["gK"] * (Vg - p["VK"]))
        a.plot(Vg, wV, "k-", lw=1.0, label=r"$dV/dt=0$"); a.plot(Vg, winf(Vg, p), "k--", lw=1.0, label=r"$dw/dt=0$")
        a.plot(D[f"trV_{c}_0"], D[f"trw_{c}_0"], color=C_SUB, lw=1.3)
        a.plot(D[f"trV_{c}_1"], D[f"trw_{c}_1"], color=C_SUP, lw=1.1)
        V0 = float(D[f"V0_{c}"]); a.plot(V0, winf(V0, p), "ko", ms=3.5, zorder=5)
        a.set_xlim(-80, 50); a.set_ylim(-0.05, 0.9); a.set_xlabel(r"$V$, мВ"); a.set_ylabel(r"$w$")
        fmt_axes(a); panel(a, f"{lab}) класс {c}")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.04, 1.0), borderaxespad=0.2)
    a = ax[2]
    for c in SETS:
        I, s = D[f"eqI_{c}"], D[f"eqS_{c}"].astype(bool); ok = (I >= 0) & (I <= 130)
        edges = np.flatnonzero(np.diff(s.astype(int)) | np.diff(ok.astype(int))) + 1
        for seg in np.split(np.arange(I.size), edges):
            if ok[seg[0]]:
                seg = np.r_[seg, min(seg[-1] + 1, I.size - 1)]          # стык с соседним участком
                a.plot(I[seg], D["eqV"][seg], color=COL[c], ls="-" if s[seg[0]] else "--",
                       lw=1.5 if s[seg[0]] else 0.9)
    a.axvspan(*HYST, color=COL["II"], alpha=0.35, lw=0)
    for x, t, dx, ha in ((ISN, r"$I_{\rm SN}$", 3, "left"), (IH, r"$I_{\rm H}$", 3, "left")):
        a.axvline(x, color="k", ls=":", lw=0.7); a.text(x + dx, 23.5, t, ha=ha, va="top", fontsize=FS)
    a.set_xlim(-4, 134); a.set_ylim(-70, 26); a.set_xlabel(r"$I$, мкА/см$^2$"); a.set_ylabel(r"$V^*$, мВ")
    a.legend(handles=[Line2D([], [], color=COL[c], lw=1.5, label=f"класс {c}") for c in SETS], loc="lower right")
    fmt_axes(a); panel(a, "в)")
    save(fig, "fig1")

    # --- Рис. 2
    fig, ax = plt.subplots(1, 3, figsize=(W, H))
    a = ax[0]
    for c in SETS:
        a.loglog(TAUS, D[f"A_{c}"], "-", marker=MRK[c], ms=3, color=COL[c], label=f"класс {c}")
        a.plot([14, 130], [D[f"A_{c}"][-1]] * 2, color=COL[c], ls=":", lw=0.8)      # уровень A_th(100 мс)
    a.loglog(TAUS, 910 / TAUS, "k--", lw=0.7, label=r"$A\tau=\mathrm{const}$")
    a.set_xlim(0.55, 130); a.set_ylim(7, 1800)
    a.set_xlabel(r"$\tau$, мс"); a.set_ylabel(r"$A_{\rm th}$, мкА/см$^2$"); a.legend(loc="lower left")
    fmt_axes(a, xt=[1, 10, 100], yt=[10, 100, 1000]); panel(a, "а)")
    a = ax[1]
    for c in SETS:
        th, bl, um = D[f"G_{c}"], D[f"B_{c}"], D[f"M_{c}"]; m = ~np.isnan(th); mb = np.isfinite(um) & m
        a.loglog(TAUS[m], th[m], "-", marker=MRK[c], ms=3, color=COL[c], label=f"класс {c}")
        a.loglog(TAUS[mb], um[mb], "s--", ms=3, mfc="w", mew=0.8, color=COL[c], lw=0.8)
        ms_ = mb & (bl > 1.2 * um)                                   # отдельный интервал спайка выше основного окна
        a.loglog(TAUS[ms_], bl[ms_], "x", ms=4, mew=0.9, color=COL[c])
    a.axhline(GSN, color="k", ls=":", lw=0.7); a.text(0.75, GSN * 1.06, r"$G_{\rm SN}$", va="bottom", fontsize=FS)
    a.set_xlim(0.55, 130); a.set_ylim(1.2, 60); a.set_xlabel(r"$\tau$, мс"); a.set_ylabel(r"$G_{\rm th}$, мСм/см$^2$")
    a.legend(loc="upper right"); fmt_axes(a, xt=[1, 10, 100], yt=[2, 5, 10, 20, 50]); panel(a, "б)")
    a = ax[2]
    im = a.pcolormesh(D["tm"], D["gm"], D["vmap"].T, cmap="viridis", shading="nearest", vmin=-30, vmax=45,
                      rasterized=True)
    for lw, col in ((1.7, "k"), (0.8, "w")):                      # белая линия с тёмной окантовкой
        a.contour(D["tm"], D["gm"], D["vmap"].T, levels=[VTH], colors=col, linewidths=lw)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel(r"$\tau$, мс"); a.set_ylabel(r"$G$, мСм/см$^2$")
    cb = fig.colorbar(im, ax=a, pad=0.03); cb.set_label(r"$V_{\max}$, мВ"); cb.ax.yaxis.set_major_formatter(FMT)
    fmt_axes(a, xt=[1, 10, 100], yt=[0.5, 1, 2, 5, 10, 20, 50]); panel(a, "в)")
    save(fig, "fig2")

    # --- Рис. 3
    fig, ax = plt.subplots(1, 3, figsize=(W, H))
    a = ax[0]
    vv = np.linspace(-5, 80, 200)
    for c in SETS:
        a.plot(vv, fit_fn(vv, *D[f"fit_{c}"]), "-", color=COL[c], lw=0.9)
        a.plot(VREVS, D[f"thG_{c}"], MRK[c], ms=3, color=COL[c], label=f"класс {c}")
    a.set_xlabel(r"$V_{\rm rev}$, мВ"); a.set_ylabel(r"$G_{\rm th}$, мСм/см$^2$"); a.legend(loc="upper right")
    fmt_axes(a); panel(a, "а)")
    a = ax[1]
    for c in SETS:
        V0 = float(D[f"V0_{c}"]); tG, tA = D[f"G_{c}"], D[f"A_{c}"]; m = ~np.isnan(tG)
        a.semilogx(TAUS[m], tG[m] * (VREV - V0) / tA[m], "-", marker=MRK[c], ms=3, color=COL[c], label=f"класс {c}")
        a.axhline((VREV - V0) / (VREV - D[f"fit_{c}"][1]), color=COL[c], ls=":", lw=0.8)
    a.set_xlabel(r"$\tau$, мс"); a.set_ylabel(r"$r=G_{\rm th}(V_{\rm rev}-V_0)/A_{\rm th}$")
    a.legend(loc="upper left"); fmt_axes(a, xt=[1, 10, 100], ydec=1); panel(a, "б)")
    a = ax[2]
    a.plot(IB_UP, D["a_up"], "o-", ms=3, color=C_SUB, label=r"$I\uparrow$")
    a.plot(IB_DN, D["a_dn"], "s-", ms=3.4, mfc="none", mew=0.8, color=C_SUP, label=r"$I\downarrow$")
    a.axvline(IH, color="k", ls=":", lw=0.7); a.text(IH - 0.25, 62, r"$I_{\rm H}$", ha="right", va="center", fontsize=FS)
    a.set_xlim(83.4, 98.6); a.set_ylim(-4, 90); a.set_xlabel(r"$I$, мкА/см$^2$"); a.set_ylabel(r"$\Delta V$, мВ")
    a.legend(loc="center", bbox_to_anchor=(0.5, 0.36), fontsize=FS, handlelength=2.0)
    fmt_axes(a); panel(a, "в)")
    save(fig, "fig3")

def main():
    cache = f"{OUT}/cache.npz"
    if "--replot" in sys.argv and os.path.exists(cache):
        D = dict(np.load(cache))
    else:
        D = compute(); np.savez(cache, **D)
    report(D); plot(D)

if __name__ == "__main__":
    main()
