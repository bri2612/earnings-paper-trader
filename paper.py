"""Live PAPER trading of several pre-registered post-earnings strategies, side by side.
No real orders are ever placed.

Every strategy uses only information known at the close of the reaction day (the first session
after the report). Each one buys at the NEXT open and sells at the close k sessions after the
reaction day. Costs are modelled for a UK retail investor: about 0.15% FX each way = 0.30% round trip.
Every trade is compared with putting the same money in SPY (S&P 500) over the same dates.

  python paper.py daily   # run once per weekday after the US close: update data, log, score, report
  python paper.py score   # just re-score the log and rewrite REPORT.md

Rules for honesty:
- Strategies are frozen once added. New ideas get a NEW letter and only count from their start date.
- Only trades logged BEFORE their entry open count ("forward" column).
- Because several strategies compete, the go-live bar includes a t-stat >= 2.5 (not just "positive").
GO-LIVE TEST per strategy (forward, closed trades only):
  >= 100 trades; average excess over the S&P 500 > +0.5%/trade; t-stat of excess >= 2.5;
  worst drawdown of a 5 x £200 account better than -20%.
"""
import sys, time, datetime as dt
from zoneinfo import ZoneInfo
import pandas as pd, numpy as np, yfinance as yf
from events import D, build

COST = 0.003
STAKE = 200
SLICE = 10  # £500 plan: £10 per trade, rest kept in an S&P 500 fund
LOG = D / "paper_trades.csv"
ET = ZoneInfo("America/New_York")

A = lambda d: (d.surp > 5) & (d.day0 > .03)
# letter: (description, signal function on the events table, holding period in sessions, start date)
STRATEGIES = {
    "A": ("Beat >5% and up >3% vs S&P on the day; hold 20 days", A, 20, "2026-10-27"),
    "B": ("Same as A; hold 5 days", A, 5, "2026-10-30"),
    "C": ("Same as A; hold 60 days", A, 60, "2026-10-30"),
    "D": ("Strong: beat >10% and up >5%; hold 20 days",
          lambda d: (d.surp > 10) & (d.day0 > .05), 20, "2026-10-30"),
    "E": ("A plus volume > 3x normal; hold 20 days",
          lambda d: A(d) & (d.volu0 > 3), 20, "2026-10-30"),
    "F": ("A plus closed above its open (no fade); hold 20 days",
          lambda d: A(d) & (d.intraday0 > 0), 20, "2026-10-30"),
    "G": ("A plus within 5% of 1-year high; hold 20 days",
          lambda d: A(d) & (d.dist_hi > -.05), 20, "2026-10-30"),
    "H": ("Bounce: beat estimates but fell >5% vs S&P; hold 20 days",
          lambda d: (d.surp > 0) & (d.day0 < -.05), 20, "2026-10-30"),
    "I": ("A plus repeat beater (beat in >=75% of last 8 reports and last one); hold 20 days",
          lambda d: A(d) & (d.beat_rate8 >= .75) & (d.last_surp > 0), 20, "2026-11-06"),
    "J": ("A plus momentum (beat the S&P over the past year); hold 20 days",
          lambda d: A(d) & (d.ret250 > 0), 20, "2026-11-13"),
}


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
    # most recent reporters first, so tonight's signals are captured even if time runs out
    syms = list(dict.fromkeys(due.assign(_t=ts[due.index]).sort_values("_t", ascending=False).sym))
    print("refreshing earnings for", len(syms), "companies")
    fresh = []
    start, budget, skipped = time.time(), 420, []
    primary_ok = _primary_works(syms[0]) if syms else False
    if not primary_ok:
        print("earnings page blocked; using the earnings-history feed as a backup")
    for s in syms:
        if time.time() - start > budget:
            skipped.append(s); continue
        if not primary_ok:
            try:
                rows = _from_history(s, due[due.sym == s])
                if rows is not None: fresh.append(rows)
                else: skipped.append(s)
            except Exception as ex:
                print(s, "backup failed", str(ex)[:80]); skipped.append(s)
            time.sleep(0.5)
            continue
        for attempt in range(2):
            try:
                e = yf.Ticker(s).get_earnings_dates(limit=12)
                if e is not None:
                    e = e.reset_index(); e["sym"] = s; fresh.append(e)
                break
            except Exception as ex:
                print(s, "retry", attempt, str(ex)[:80])
                if attempt == 1: skipped.append(s)
                else: time.sleep(15)
        time.sleep(0.7)
    print(f"earnings refreshed: {len(fresh)}, skipped (time limit or Yahoo refusal): {len(skipped)}",
          ("e.g. " + ", ".join(skipped[:10])) if skipped else "")
    if fresh:
        all_ = pd.concat([pd.concat(fresh), ed])  # new rows first so they win on duplicates
        all_["_k"] = pd.to_datetime(all_["Earnings Date"], utc=True).dt.normalize()
        all_ = all_.drop_duplicates(["sym", "_k"]).drop(columns="_k")
        all_.reset_index(drop=True).to_pickle(D / "earnings.pkl")


def _primary_works(sym):
    try:
        yf.Ticker(sym).get_earnings_dates(limit=4); return True
    except Exception as ex:
        print("earnings page check failed:", str(ex)[:80]); return False


def _from_history(sym, due_rows):
    """Backup source: fill actual EPS, estimate and surprise for scheduled reports from Yahoo's
    earnings-history feed (last 4 fiscal quarters). The report date and time come from the
    schedule we already hold; a quarter is matched only if it ended 0-120 days before the report."""
    eh = yf.Ticker(sym).earnings_history
    if eh is None or len(eh) == 0: return None
    eh = eh.copy(); eh.index = pd.to_datetime(eh.index)
    out = []
    for _, r in due_rows.iterrows():
        rd = pd.to_datetime(r["Earnings Date"], utc=True).tz_convert(None)
        q = eh[(eh.index <= rd) & (eh.index >= rd - pd.Timedelta(days=120))]
        if len(q) == 0 or pd.isna(q.iloc[-1].epsActual): continue
        row = r.copy(); x = q.iloc[-1]
        row["EPS Estimate"], row["Reported EPS"] = x.epsEstimate, x.epsActual
        row["Surprise(%)"] = x.surprisePercent * 100
        out.append(row)
    return pd.DataFrame(out) if out else None


def is_forward(rdate, logged_at_et):
    """Genuine forward test = logged before the entry (next weekday's 9:30 ET open)."""
    entry = (pd.Timestamp(rdate) + pd.offsets.BDay(1)).replace(hour=9, minute=30)
    return pd.Timestamp(str(logged_at_et)[:16].replace("T", " ")) < entry


def log_signals():
    ev = pd.read_pickle(D / "events.pkl")
    recent = ev[ev.rdate >= ev.rdate.max() - pd.Timedelta(days=7)]
    now = dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M")
    new = []
    for k, (_, f, hold, start) in STRATEGIES.items():
        s = recent[f(recent) & (recent.rdate >= pd.Timestamp(start))]
        s = s[["sym", "rdate", "surp", "day0"]].assign(strategy=k, hold=hold, logged_at_et=now)
        new.append(s)
    sig = pd.concat(new)
    if LOG.exists():
        old = pd.read_csv(LOG, parse_dates=["rdate"])
        if "strategy" not in old: old = old.assign(strategy="A", hold=20)
        seen = set(zip(old.strategy, old.sym, old.rdate.dt.date))
        sig = sig[[(k, s, d.date()) not in seen for k, s, d in zip(sig.strategy, sig.sym, sig.rdate)]]
        sig = pd.concat([old, sig])
    sig["forward"] = [is_forward(r, l) for r, l in zip(sig.rdate, sig.logged_at_et)]
    sig.to_csv(LOG, index=False)
    return sig


def score(write=True):
    log = pd.read_csv(LOG, parse_dates=["rdate"])
    if "strategy" not in log: log = log.assign(strategy="A", hold=20)
    px = pd.read_pickle(D / "prices_long.pkl"); spy = px["SPY"]
    rows = []
    for _, r in log.iterrows():
        row = r.to_dict(); hold = int(r.hold)
        p = px[r.sym].dropna(subset=["Close"]); i = p.index.searchsorted(r.rdate)
        if i + 1 >= len(p):
            row.update(status="waiting for entry"); rows.append(row); continue
        e, x = i + 1, min(i + hold, len(p) - 1)
        stk = p.Close.iloc[x] / p.Open.iloc[e] - 1 - COST
        mkt = spy.Close.loc[p.index[x]] / spy.Open.loc[p.index[e]] - 1 - COST
        row.update(entry=p.index[e].date(), exit_or_mark=p.index[x].date(), held=x - i,
                   status="closed" if x == i + hold else "open",
                   ret_pct=round(stk * 100, 2), spy_pct=round(mkt * 100, 2),
                   excess_pct=round((stk - mkt) * 100, 2), pnl_gbp=round(STAKE * stk, 2))
        rows.append(row)
    out = pd.DataFrame(rows)
    if write: out.to_csv(D / "paper_scored.csv", index=False)
    return out


def stats(c):
    n = len(c)
    if n == 0: return dict(n=0)
    eq = c.sort_values("exit_or_mark").pnl_gbp.cumsum()
    t = c.excess_pct.mean() / c.excess_pct.std() * np.sqrt(n) if n > 2 and c.excess_pct.std() > 0 else float("nan")
    return dict(n=n, win=np.mean(c.ret_pct > 0), avg=c.ret_pct.mean(), spy=c.spy_pct.mean(),
                ex=c.excess_pct.mean(), t=t, pnl=c.pnl_gbp.sum(),
                dd=(eq - eq.cummax()).min() / (STAKE * 5) * 100)


def report(out):
    if "status" not in out: out = out.assign(status="waiting for entry")
    L = [f"# Paper-trading report — {dt.date.today()}", "",
         "Paper only. Genuine forward trades only (logged before entry). UK costs of 0.3% included.", "",
         "| | Strategy | Closed | Win | Avg | S&P 500 same dates | Beat S&P by | t | £200/trade P&L | £500 plan: extra vs S&P | Open | Verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    best = None
    for k, (desc, _, hold, start) in STRATEGIES.items():
        sub = out[(out.strategy == k) & (out.forward == True)]
        s = stats(sub[sub.status == "closed"]); n_open = (sub.status == "open").sum()
        go = s["n"] >= 100 and s["ex"] > 0.5 and s["t"] >= 2.5 and s["dd"] > -20
        if go: best = best or k
        if s["n"]:
            L.append(f"| {k} | {desc} | {s['n']} | {s['win']:.0%} | {s['avg']:+.2f}% | {s['spy']:+.2f}% | "
                     f"{s['ex']:+.2f}% | {s['t']:.1f} | £{s['pnl']:+.0f} | £{s['n'] * SLICE * s['ex'] / 100:+.2f} | {n_open} | {'**GO**' if go else 'no-go'} |")
        else:
            L.append(f"| {k} | {desc} | 0 | | | | | | | | {n_open} | no-go |")
    L += ["", "Go-live test per strategy: ≥100 closed forward trades, beats the S&P 500 by >0.5%/trade, "
          "t ≥ 2.5, worst drawdown better than −20%.",
          "£500 plan = keep £500 in an S&P 500 fund and move £10 into each signal; the column shows how much "
          "more (or less) that would have made than just holding the fund.", "",
          f"**Verdict: {'GO — strategy ' + best + ' passed; start the £500 plan: £500 in an S&P 500 fund, move £10 into each signal' if best else 'NO-GO — keep paper trading'}**", ""]

    new = out[out.status == "waiting for entry"]
    L.append("## BUY at the next US open (2:30pm UK) — paper")
    if len(new):
        for sym, g in new.groupby("sym"):
            r = g.iloc[0]
            L.append(f"- {sym} (strategies {','.join(sorted(g.strategy))}): beat by {r.surp:.1f}%, "
                     f"{r.day0 * 100:+.1f}% vs S&P on report day")
    else:
        L.append("- none")
    L.append("")
    sell = out[(out.status == "open") & (out.held == out.hold - 1)]
    L.append("## SELL on the next trading day, any time before the US close (9pm UK) — paper")
    L += [f"- {r.sym} (strategy {r.strategy}, bought {r.entry}, now {r.ret_pct:+.1f}%)" for r in sell.itertuples()] or ["- none"]
    txt = "\n".join(L)
    (D.parent / "REPORT.md").write_text(txt)
    print(txt)


def daily():
    update_prices(); update_earnings(); build(); log_signals(); report(score())


if __name__ == "__main__":
    {"daily": daily, "score": lambda: report(score()),
     "log": lambda: (log_signals(), report(score()))}[sys.argv[1]]()
