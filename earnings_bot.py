"""Earnings-reaction predictor + paper trader for S&P 500 stocks.

Usage:
  python earnings_bot.py fetch        # download S&P 500 list, earnings history, prices
  python earnings_bot.py backtest     # walk-forward train/test, honest out-of-sample report
  python earnings_bot.py predict      # log predictions for upcoming earnings (paper trades)
  python earnings_bot.py score        # score past paper trades against actual moves

Every feature uses only information available BEFORE the earnings report.
"""
import sys, json, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier

D = Path(__file__).parent / "data"; D.mkdir(exist_ok=True)
COST = 0.001  # 10 bps round-trip trading cost assumption
CONF = 0.55   # only trade when model probability is this far from 50/50

def sp500():
    import requests, io
    h = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                     headers={"User-Agent": "Mozilla/5.0 earnings-research"}, timeout=30).text
    t = pd.read_html(io.StringIO(h))[0]
    t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    t[["Symbol", "GICS Sector"]].to_csv(D / "sp500.csv", index=False)
    return t

def fetch():
    tick = sp500()
    px = yf.download(list(tick.Symbol) + ["SPY"], period="7y", auto_adjust=True,
                     group_by="ticker", threads=True, progress=False)
    px.to_pickle(D / "prices.pkl")
    rows = []
    for i, s in enumerate(tick.Symbol):
        for attempt in range(3):
            try:
                e = yf.Ticker(s).get_earnings_dates(limit=32)
                if e is not None:
                    e = e.reset_index(); e["sym"] = s; rows.append(e)
                break
            except Exception as ex:
                print(s, "retry", attempt, str(ex)[:80]); time.sleep(10 * (attempt + 1))
        time.sleep(0.7)
        if i % 50 == 0: print(f"{i}/{len(tick)}"); time.sleep(1)
    ed = pd.concat(rows); ed.to_pickle(D / "earnings.pkl")
    print("earnings rows:", len(ed), "companies:", ed.sym.nunique())

def build():
    px = pd.read_pickle(D / "prices.pkl"); ed = pd.read_pickle(D / "earnings.pkl")
    sec = pd.read_csv(D / "sp500.csv").set_index("Symbol")["GICS Sector"]
    ed = ed.rename(columns={"Earnings Date": "date", "EPS Estimate": "eps_est",
                            "Reported EPS": "eps_act", "Surprise(%)": "surprise"})
    ed["ts"] = pd.to_datetime(ed["date"], utc=True).dt.tz_convert("America/New_York")
    spy = px["SPY"]["Close"]
    out = []
    for s, g in ed.groupby("sym"):
        if s not in px.columns.get_level_values(0): continue
        p = px[s].dropna()
        if len(p) < 300: continue
        c, o, idx = p["Close"], p["Open"], p.index
        g = g.sort_values("ts")
        hist_beats = []
        for _, r in g.iterrows():
            # after-close report (>=16:00) reacts next session; pre-market reacts same day
            day = r.ts.tz_localize(None).normalize()
            amc = r.ts.hour >= 12
            pos = idx.searchsorted(day + pd.Timedelta(days=1 if amc else 0))
            if pos < 60: continue
            pre = pos - 1  # last close strictly before the reaction
            f = dict(sym=s, date=idx[pos] if pos < len(idx) else day, sector=sec.get(s, "NA"),
                     ret5=c.iloc[pre] / c.iloc[pre - 5] - 1,
                     ret20=c.iloc[pre] / c.iloc[pre - 20] - 1,
                     ret60=c.iloc[pre] / c.iloc[pre - 60] - 1,
                     rel20=(c.iloc[pre] / c.iloc[pre - 20]) - (spy.asof(idx[pre]) / spy.asof(idx[pre - 20])),
                     vol20=c.pct_change().iloc[pre - 20:pre].std(),
                     dist_hi=c.iloc[pre] / c.iloc[pre - 252:pre].max() - 1 if pre > 252 else np.nan,
                     beat_rate=np.mean(hist_beats[-8:]) if hist_beats else np.nan,
                     n_hist=len(hist_beats),
                     amc=int(amc), month=day.month)
            reacted = pos < len(idx) and not np.isnan(r.get("eps_act", np.nan))
            if reacted:
                f["y_ret"] = (c.iloc[pos] / c.iloc[pre] - 1) - (spy.asof(idx[pos]) / spy.asof(idx[pre]) - 1)
                hist_beats.append(float(r.eps_act > r.eps_est) if not np.isnan(r.eps_est) else 0.5)
            else:
                f["y_ret"] = np.nan
            out.append(f)
    df = pd.DataFrame(out)
    df["sector"] = df["sector"].astype("category")
    return df

FEATS = ["ret5", "ret20", "ret60", "rel20", "vol20", "dist_hi", "beat_rate", "n_hist", "amc", "month", "sector"]

def model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=50, categorical_features="from_dtype")

def backtest():
    df = build().dropna(subset=["y_ret"]).sort_values("date")
    df["y"] = (df.y_ret > 0).astype(int)
    qs = pd.PeriodIndex(df.date, freq="Q"); uq = sorted(qs.unique())
    res = []
    for q in uq[8:]:  # need 2 years of history before first test quarter
        tr, te = df[qs < q - 1], df[qs == q]  # 1-quarter gap to avoid leakage
        if len(te) == 0: continue
        m = model().fit(tr[FEATS], tr.y)
        te = te.assign(p=m.predict_proba(te[FEATS])[:, 1]); res.append(te)
    r = pd.concat(res)
    r["hit"] = ((r.p > .5) == (r.y == 1))
    t = r[(r.p > CONF) | (r.p < 1 - CONF)].copy()
    t["pnl"] = np.where(t.p > .5, t.y_ret, -t.y_ret) - COST
    base = max(r.y.mean(), 1 - r.y.mean())
    rep = {
        "test_events": len(r), "accuracy_all": round(r.hit.mean(), 4),
        "always_majority_baseline": round(base, 4),
        "trades_taken": len(t), "accuracy_traded": round(((t.p > .5) == (t.y == 1)).mean(), 4),
        "avg_pnl_per_trade_pct": round(t.pnl.mean() * 100, 3),
        "total_pnl_pct_equal_weight": round(t.pnl.sum() * 100, 1),
        "worst_trade_pct": round(t.pnl.min() * 100, 2),
        "quarterly_accuracy": r.groupby(pd.PeriodIndex(r.date, freq="Q").astype(str)).hit.mean().round(3).to_dict(),
    }
    print(json.dumps(rep, indent=2)); (D / "backtest_report.json").write_text(json.dumps(rep, indent=2))
    r.to_csv(D / "backtest_predictions.csv", index=False)

def predict():
    df = build().sort_values("date"); hist = df.dropna(subset=["y_ret"])
    m = model().fit(hist[FEATS], (hist.y_ret > 0).astype(int))
    today = pd.Timestamp.today().normalize()
    up = df[df.y_ret.isna() & (df.date >= today) & (df.date <= today + pd.Timedelta(days=14))].copy()
    up["p_up"] = m.predict_proba(up[FEATS])[:, 1]
    up = up[(up.p_up > CONF) | (up.p_up < 1 - CONF)]
    up["call"] = np.where(up.p_up > .5, "LONG", "SHORT"); up["logged_at"] = dt.datetime.now().isoformat()
    log = D / "paper_trades.csv"
    up[["logged_at", "sym", "date", "call", "p_up"]].to_csv(log, mode="a", header=not log.exists(), index=False)
    print(up[["sym", "date", "call", "p_up"]].to_string(index=False))

def score():
    log = pd.read_csv(D / "paper_trades.csv", parse_dates=["date"]).drop_duplicates(["sym", "date"])
    act = build().dropna(subset=["y_ret"])[["sym", "date", "y_ret"]]
    m = log.merge(act, on=["sym", "date"])
    m["pnl"] = np.where(m.call == "LONG", m.y_ret, -m.y_ret) - COST
    print(m.to_string(index=False))
    print(f"\nScored {len(m)}  hit rate {(m.pnl + COST > 0).mean():.1%}  avg pnl {m.pnl.mean():.2%}")

if __name__ == "__main__":
    {"fetch": fetch, "backtest": backtest, "predict": predict, "score": score}[sys.argv[1]]()
