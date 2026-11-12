#!/usr/bin/env python3
"""
Контроль численных результатов к ml_figures.py (модель Морриса–Лекара).
Запуск:  python ml_checks.py     (после python ml_figures.py; около 2–3 мин)
Выход:   output/checks_ML.csv, сводка в stdout.
Проверки: (а) шаг интегрирования dt = 0.1 и 0.05 мс; (б) время наблюдения Tpost = 100, 150, 300 мс;
          (в) верхние границы спайковых окон на сетке по G в 2 раза мельче и связность окон;
          (г) границы гистерезиса класса II с шагом по току 0.1; (д) статические оценки из текста;
          (е) пороги класса I при tau = 200, 400, 800 мс.
Физические параметры, критерий спайка (V_max > +20 мВ) и V_rev берутся из ml_figures.py без изменений.
"""
import numpy as np
import pandas as pd
import ml_figures as m

TAUS, SETS, OUT = m.TAUS, m.SETS, m.OUT
D = dict(np.load(f"{OUT}/cache.npz"))            # базовый расчёт: dt = 0.1 мс, Tpost = 150 мс
rows = []

def thr(dt, Tpost):
    return {(c, mode): m.threshold(SETS[c], mode, TAUS, dt=dt, Tpost=Tpost)[0] for c in SETS for mode in "IG"}

def compare(name, new):
    for (c, mode), v in new.items():
        base = D[f"{'A' if mode == 'I' else 'G'}_{c}"]
        same = np.array_equal(np.isnan(base), np.isnan(v)); ok = ~np.isnan(base) & ~np.isnan(v)
        d = 100 * (v[ok] / base[ok] - 1); i = np.argmax(np.abs(d))
        print(f"  {name}: класс {c}, {'A_th' if mode == 'I' else 'G_th'}: макс. отличие {d[i]:+.4f}% "
              f"(tau = {TAUS[ok][i]:g} мс); набор tau со спайком {'тот же' if same else 'ИЗМЕНИЛСЯ'}")
        for t, b, x in zip(TAUS, base, v):
            rows.append(dict(проверка=name, класс=c, величина="A_th" if mode == "I" else "G_th", tau_ms=t,
                             базовое=b, контрольное=x, отличие_проц=100 * (x / b - 1)))

print("(а) шаг интегрирования: dt = 0.05 мс против базового 0.1 мс (Tpost = 150 мс)")
compare("dt=0.05", thr(0.05, 150.))
print("(б) время наблюдения после импульса: Tpost = 100 и 300 мс против базового 150 мс (dt = 0.1 мс)")
for Tp in (100., 300.):
    compare(f"Tpost={Tp:g}", thr(m.DT, Tp))

print("(в) верхние границы спайковых окон: сетка по G из 199 узлов против базовой из 100 (0.05…80 мСм/см2)")
g2 = np.geomspace(0.05, 80, 199)                 # каждый второй узел совпадает с базовой сеткой
for c, p in SETS.items():
    vm = m.sim(p, "G", np.repeat(g2[None, :], TAUS.size, 0).ravel(),
               np.repeat(TAUS[:, None], g2.size, 1).ravel()).reshape(TAUS.size, g2.size)
    for i, t in enumerate(TAUS):
        sp = vm[i] > m.VTH
        if not sp.any():
            print(f"  класс {c}, tau = {t:g} мс: спайка нет во всём диапазоне G"); continue
        e = np.flatnonzero(np.diff(np.r_[0, sp.astype(int), 0]))      # начала и концы спайковых интервалов
        wins = [(g2[a], g2[b - 1], g2[b] if b < g2.size else np.inf) for a, b in zip(e[::2], e[1::2])]
        base = D[f"B_{c}"][i]; top = wins[-1][2]
        if np.isinf(top) and np.isinf(base) and len(wins) == 1:
            continue                                                  # верхней границы нет на обеих сетках
        print(f"  класс {c}, tau = {t:g} мс: верхняя граница {base:.2f} -> {top:.2f} мСм/см2 "
              f"({100 * (top / base - 1):+.1f}%); интервалов со спайком: {len(wins)}; "
              + "; ".join(f"[{a:.2f}; {b:.2f}], первый узел без спайка {x:.2f}" for a, b, x in wins))
        rows.append(dict(проверка="сетка G x2", класс=c, величина="G_block", tau_ms=t, базовое=base,
                         контрольное=top, отличие_проц=100 * (top / base - 1), интервалов=len(wins),
                         граница_основного_окна=wins[0][2]))

print("(г) гистерезис класса II, шаг по току 0.1 мкА/см2 (800 мс на шаг, как в основном расчёте)")
p = SETS["II"]
Idn = np.r_[98., 95., 92., 90., np.arange(89.0, 87.75, -0.1)]; adn = m.sweep(p, Idn, -20., 0.2)
V0 = m.rest(84., p)[0]
Iup = np.r_[84., 88., 92., np.arange(93.0, 95.05, 0.1)]; aup = m.sweep(p, Iup, V0, m.winf(V0, p))
i = np.argmax(adn < 40); j = np.argmax(aup > 40)
print(f"  срыв колебаний при убывании тока: между {Idn[i]:.1f} и {Idn[i - 1]:.1f}; "
      f"возникновение при росте тока: между {Iup[j - 1]:.1f} и {Iup[j]:.1f} мкА/см2")
print("  размах колебаний, мВ: " + ", ".join(f"I={x:.1f}: {a:.1f}" for x, a in zip(Idn[:i], adn[:i])))

print("(д) статические оценки и производные величины")
Gs = np.linspace(0, 10, 201); tr = []
for G in Gs:                                     # класс II: равновесие под кондуктансной нагрузкой, V_rev = 0
    f = lambda V: m.Iion(V, m.winf(V, p), p) + G * (m.VREV - V)
    Vs = np.linspace(-80, 60, 5601); fs = f(Vs)
    roots = [m.brentq(f, Vs[k], Vs[k + 1]) for k in range(Vs.size - 1) if fs[k] * fs[k + 1] < 0]
    J = m.jac(roots[0], G * (m.VREV - roots[0]), p); J[0, 0] -= G / p["C"]
    tr.append((len(roots), np.trace(J), np.linalg.det(J)))
tr = np.array(tr)
print(f"  класс II, G = 0…10 мСм/см2: число равновесий {int(tr[:, 0].min())}…{int(tr[:, 0].max())}; "
      f"max tr J = {tr[:, 1].max():+.4f} 1/мс; min det J = {tr[:, 2].min():+.5f}")
def chron(tau, th):                              # хронаксия: th = 2 th(100 мс), линейная интерполяция в log–log
    y = np.log(th / (2 * th[-1])); k = np.argmax(y < 0)
    return np.exp(np.interp(0, [y[k], y[k - 1]], [np.log(tau[k]), np.log(tau[k - 1])]))
print(f"  хронаксия по току: класс I {chron(TAUS, D['A_I']):.1f} мс, класс II {chron(TAUS, D['A_II']):.1f} мс; "
      f"по проводимости (класс I): {chron(TAUS, D['G_I']):.1f} мс")
sel = TAUS >= 1.5
for c in SETS:
    A = D[f"A_{c}"][sel]; t = TAUS[sel]
    k1 = np.polyfit(t, A * t, 1)                                        # Вайс: A*tau = A_rh*tau + Q0
    k2 = np.polyfit(1 / t, A, 1)                                        # Лапик: A = A_rh + Q0/tau
    k3 = np.polyfit(1 / t, A, 1, w=1 / A)                               # то же, относительные отклонения
    print(f"  Вайс–Лапик (tau >= 1.5 мс), класс {c}: A_rh = {k1[0]:.1f} (МНК по A*tau), {k2[1]:.1f} (МНК по A), "
          f"{k3[1]:.1f} (МНК по относительным отклонениям)")
print(f"  A_th(100 мс)/I_SN - 1 = {100 * (D['A_I'][-1] / D['ISN'] - 1):.1f}%; "
      f"G_th(100 мс)/G_SN - 1 = {100 * (D['G_I'][-1] / D['GSN'] - 1):.1f}%; "
      f"A_th: 60 -> 100 мс: {100 * (D['A_I'][-1] / D['A_I'][-2] - 1):+.1f}% (кл. I), "
      f"{100 * (D['A_II'][-1] / D['A_II'][-2] - 1):+.2f}% (кл. II); "
      f"G_th: 60 -> 100 мс: {100 * (D['G_I'][-1] / D['G_I'][-2] - 1):+.1f}% (кл. I)")
pd.DataFrame(rows).to_csv(f"{OUT}/checks_ML.csv", index=False)

print("(е) класс I при длинных импульсах: стремление порога к статическим значениям I_SN и G_SN")
tl = [200., 400., 800.]
Al = m.threshold(SETS["I"], "I", tl)[0]; Gl, Bl = m.threshold(SETS["I"], "G", tl)
for k, x in enumerate(tl):
    print(f"  класс I, tau = {x:g} мс: A_th = {Al[k]:.2f} ({100 * (Al[k] / D['ISN'] - 1):+.1f}% к I_SN); "
          f"G_th = {Gl[k]:.3f} ({100 * (Gl[k] / D['GSN'] - 1):+.1f}% к G_SN); G_block = {Bl[k]:.2f}")
