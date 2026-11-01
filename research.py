"""Backtest a candidate post-earnings strategy the same way every time, before it is pre-registered.

  python research.py            # re-check all strategies in paper.STRATEGIES
Use from code:  from research import evaluate; evaluate(lambda d: (d.surp > 5) & (d.day0 > .03), 20)

Needs data/events.pkl and data/prices_long.pkl (create with:
  python -c "import paper, events; paper.update_prices(); events.build()").
Signals may only use columns known at the reaction-day close: surp, est, act, day0, day0_z, gap,
intraday0, volu0, and all pre-report columns (ret5, ret20, ret60, ret250, vol20, vol250, dist_hi,
volu_pre, beat_rate8, last_surp, med_surp4, last_day0, avg_day0, avg_absday0, amc, sector).
Never use post* or y_* columns in a signal: they are future returns.
"""
import numpy as np, pandas as pd
from events import D

COST = 0.003
_cache = {}


def _data():
    if not _cache:
        ev = pd.read_pickle(D / "events.pkl")
        _cache["ev"] = ev[ev.rdate >= "2014-01-01"]
        _cache["px"] = pd.read_pickle(D / "prices_long.pkl")
    return _cache["ev"], _cache["px"]


def evaluate(signal, hold, name=""):
    ev, px = _data(); spy = px["SPY"]; res = {}
    for part, d in [("old", ev[~ev.holdout]), ("locked", ev[ev.holdout])]:
        out = []
        for _, r in d[signal(d)].iterrows():
            p = px[r.sym].dropna(subset=["Close"]); j = p.index.searchsorted(r.rdate)
            if j + hold >= len(p): continue
            a, b = p.index[j + 1], p.index[j + hold]
            s = p.Close.iloc[j + hold] / p.Open.iloc[j + 1] - 1 - COST
            m = spy.Close.loc[b] / spy.Open.loc[a] - 1 - COST
            out.append((s, s - m, r.rdate.year))
        r = pd.DataFrame(out, columns=["ret", "ex", "yr"])
        n = len(r)
        t = r.ex.mean() / r.ex.std() * np.sqrt(n) if n > 2 else float("nan")
        yrs = (r.groupby("yr").ex.mean() > 0)
        res[part] = dict(n=n, win=(r.ret > 0).mean() if n else np.nan, ex=r.ex.mean() * 100 if n else np.nan,
                         t=t, yrs=f"{yrs.sum()} of {len(yrs)}")
    o, l = res["old"], res["locked"]
    print(f"{name:45s} old: {o['n']} / {o['ex']:+.2f}% / t {o['t']:.1f} / {o['yrs']}   "
          f"locked: {l['n']} / {l['ex']:+.2f}% / t {l['t']:.1f}")
    return res


if __name__ == "__main__":
    from paper import STRATEGIES
    for k, (desc, f, hold, _) in STRATEGIES.items():
        evaluate(f, hold, f"{k} {desc}"[:45])
