"""Live paper trading of the FROZEN post-earnings drift rule. No real orders are ever sent.

Rule (frozen 2026-11-01, before the locked holdout test):
  After an earnings report, if EPS beat estimates by > 5% AND the stock beat SPY by > 3% on the
  reaction day -> BUY at the next day's open, SELL at the close 20 trading days after the reaction day.
  Costs modelled for a UK retail investor: ~0.15% FX each way = 0.30% round trip, no hedge.
  Every trade is compared against putting the same money in SPY (S&P 500) over the same dates.

  python paper.py daily   # run once per weekday after the US close: update data, log, score, report
  python paper.py score   # just re-score the log

Only trades logged BEFORE their entry open count as genuine forward tests ("forward" column).
GO-LIVE TEST (all on forward, closed trades):
  >= 100 trades;  average excess over SPY > +0.5% per trade;  worst drawdown of the £200-per-trade
  account better than -20%.
"""
import sys, time, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd, numpy as np, yfinance as yf
from events import D, build

COST = 0.003
STAKE = 200
LOG = D / "paper_trades.csv"
ET = ZoneInfo("America/New_York")


def update_prices():
    t = pd.read_csv(D / "sp500.csv").Symbol.tolist() + ["SPY"]
    px = yf.download(t, start="2007-01-01", auto_adjust=True, group_by="ticker",
                     threads=True, progress=False)
    px.to_pickle(D / "prices_long.pkl")
    print("prices to", px.index.max().date())


def update_earnings(days_back=6):
    """Re-fetch earnings only for companies whose scheduled report fell in the last few days."""
    ed = pd.read_pickle(D / "earnings.pkl").reset_index(drop=True)
    ts = pd.to_datetime(ed["Earnings Date"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    due = ed[(ts >= now - pd.Timedelta(days=days_back)) & (ts <= now)]
    syms = sorted(set(due.sym))
    print("refreshing earnings for", len(syms), "companies")
    fresh = []
    for s in syms:
        for attempt in range(3):
            try:
                e = yf.Ticker(s).get_earnings_dates(limit=12)
                if e is not None:
                    e = e.reset_index(); e["sym"] = s; fresh.append(e)
                break
            except Exception as ex:
                print(s, "retry", attempt, str(ex)[:80]); time.sleep(10 * (attempt + 1))
        time.sleep(0.7)
    if fresh:
        new = pd.concat(fresh)
        all_ = pd.concat([new, ed])  # new rows first so they win on duplicates
        all_["_k"] = pd.to_datetime(all_["Earnings Date"], utc=True).dt.normalize()
        all_ = all_.drop_duplicates(["sym", "_k"]).drop(columns="_k")
        all_.reset_index(drop=True).to_pickle(D / "earnings.pkl")


def log_signals():
    ev = pd.read_pickle(D / "events.pkl")
    last = ev.rdate.max()
    recent = ev[ev.rdate >= last - pd.Timedelta(days=7)]
    sig = recent[(recent.surp > 5) & (recent.day0 > .03)][["sym", "rdate", "surp", "day0"]].copy()
    now = dt.datetime.now(ET)
    sig["logged_at_et"] = now.strftime("%Y-%m-%d %H:%M")
    if LOG.exists():
        old = pd.read_csv(LOG, parse_dates=["rdate"])
        k_old = set(zip(old.sym, old.rdate.dt.date))
        sig = sig[[(s, d.date()) not in k_old for s, d in zip(sig.sym, sig.rdate)]]
        sig = pd.concat([old, sig])
    sig["forward"] = [is_forward(r, l) for r, l in zip(sig.rdate, sig.logged_at_et)]
    sig.to_csv(LOG, index=False)
    return sig


def is_forward(rdate, logged_at_et):
    """Genuine forward test = logged before the entry (next weekday's 9:30 ET open)."""
    entry = (pd.Timestamp(rdate) + pd.offsets.BDay(1)).replace(hour=9, minute=30)
    return pd.Timestamp(str(logged_at_et)[:16].replace("T", " ")) < entry


def score(write=True):
    log = pd.read_csv(LOG, parse_dates=["rdate"])
    px = pd.read_pickle(D / "prices_long.pkl"); spy = px["SPY"]
    rows = []
    for _, r in log.iterrows():
        row = r.to_dict()
        p = px[r.sym].dropna(subset=["Close"]); i = p.index.searchsorted(r.rdate)
        if i + 1 >= len(p):
            row.update(status="waiting for entry"); rows.append(row); continue
        e, x = i + 1, min(i + 20, len(p) - 1)
        stk = p.Close.iloc[x] / p.Open.iloc[e] - 1 - COST
        mkt = spy.Close.loc[p.index[x]] / spy.Open.loc[p.index[e]] - 1 - COST
        row.update(entry=p.index[e].date(), exit_or_mark=p.index[x].date(),
                   status="closed" if x == i + 20 else "open", held=x - i,
                   ret_pct=round(stk * 100, 2), spy_pct=round(mkt * 100, 2),
                   excess_pct=round((stk - mkt) * 100, 2), pnl_gbp=round(STAKE * stk, 2))
        rows.append(row)
    out = pd.DataFrame(rows)
    if write: out.to_csv(D / "paper_scored.csv", index=False)
    return out


def report(out):
    lines = [f"# Paper-trading report — {dt.date.today()}", ""]
    for label, sub in [("Genuine forward trades", out[out.forward == True]),
                       ("All logged (incl. retroactive)", out)]:
        c = sub[sub.status == "closed"]; o = sub[sub.status == "open"]
        lines.append(f"## {label}")
        if len(c):
            eq = c.sort_values("exit_or_mark").pnl_gbp.cumsum()
            dd = (eq - eq.cummax()).min()
            lines += [f"- Closed: {len(c)} | win rate {np.mean(c.ret_pct > 0):.0%} | "
                      f"avg {c.ret_pct.mean():+.2f}% vs S&P 500 {c.spy_pct.mean():+.2f}% "
                      f"(excess {c.excess_pct.mean():+.2f}%)",
                      f"- £{STAKE}/trade P&L: £{c.pnl_gbp.sum():+.0f} | S&P 500 same money: "
                      f"£{(STAKE * c.spy_pct / 100).sum():+.0f} | worst drawdown £{dd:.0f} "
                      f"({dd / (STAKE * 5) * 100:.0f}% of a 5-position £{STAKE * 5} account)"]
        else:
            lines.append("- No closed trades yet")
        lines.append(f"- Open: {len(o)} (unrealised £{o.pnl_gbp.sum():+.0f}); waiting: {(sub.status == 'waiting for entry').sum()}")
        lines.append("")
    f = out[(out.forward == True) & (out.status == "closed")]
    n = len(f); ex = f.excess_pct.mean() if n else float("nan")
    eq = f.sort_values("exit_or_mark").pnl_gbp.cumsum() if n else pd.Series([0.0])
    dd_pct = (eq - eq.cummax()).min() / (STAKE * 5) * 100
    checks = [(f"≥100 forward closed trades (have {n})", n >= 100),
              (f"avg excess over S&P 500 > +0.5% (have {ex:+.2f}%)", n > 0 and ex > 0.5),
              (f"worst drawdown better than -20% (have {dd_pct:.0f}%)", n > 0 and dd_pct > -20)]
    lines.append("## Go-live test")
    lines += [f"- {'PASS' if ok else 'not yet'}: {t}" for t, ok in checks]
    verdict = "GO — ok to start with £200 per trade" if all(ok for _, ok in checks) else "NO-GO — keep paper trading"
    lines += ["", f"**Verdict: {verdict}**", ""]
    new = out[out.status == "waiting for entry"]
    if len(new):
        lines.append("## BUY tomorrow at the US open (2:30pm UK), or queue as an order for the open")
        lines += [f"- {r.sym}: beat by {r.surp:.1f}%, beat S&P by {r.day0 * 100:.1f}% on report day" for r in new.itertuples()]
        lines.append("")
    # trades held 19 sessions so far -> their 20th (final) session is tomorrow
    sell = out[(out.status == "open") & (out.held == 19)] if "held" in out else out.iloc[0:0]
    lines.append("## SELL tomorrow, any time before the US close (9pm UK)")
    lines += [f"- {r.sym} (bought {r.entry}, currently {r.ret_pct:+.1f}%)" for r in sell.itertuples()] or ["- none"]
    txt = "\n".join(lines)
    (D.parent / "REPORT.md").write_text(txt)
    print(txt)


def daily():
    update_prices(); update_earnings(); build(); log_signals(); report(score())


if __name__ == "__main__":
    {"daily": daily, "score": lambda: report(score())}[sys.argv[1]]()
