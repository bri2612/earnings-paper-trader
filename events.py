"""Build a clean event table: one row per earnings report, with
 - features known BEFORE the report (for pre-earnings prediction)
 - facts known right AFTER the report (surprise, gap, day-0 move) for post-earnings strategies
 - forward returns, all measured relative to SPY (abnormal returns)
Timing: reports at <=9:xx ET are before-open (react same session), >=16:00 after-close
(react next session). Mid-session timestamps are ambiguous and dropped.
"""
from pathlib import Path
import numpy as np, pandas as pd

D = Path(__file__).parent / "data"
HOLDOUT_START = pd.Timestamp("2025-10-01")  # locked final exam: do not tune on anything after this


def build():
    px = pd.read_pickle(D / "prices_long.pkl")
    ed = pd.read_pickle(D / "earnings.pkl").reset_index(drop=True)
    sec = pd.read_csv(D / "sp500.csv").set_index("Symbol")["GICS Sector"]
    ed = ed.rename(columns={"Earnings Date": "date", "EPS Estimate": "est",
                            "Reported EPS": "act", "Surprise(%)": "surp"})
    ed["ts"] = pd.to_datetime(ed["date"], utc=True).dt.tz_convert("America/New_York")
    ed = ed.drop_duplicates(["sym", "ts"]).sort_values(["sym", "ts"])
    spy = px["SPY"]["Close"].dropna()
    syms = set(px.columns.get_level_values(0))
    out = []
    for s, g in ed.groupby("sym"):
        if s not in syms: continue
        p = px[s].dropna(subset=["Close"])
        p = p[p.Close > 0]
        if len(p) < 300: continue
        idx = p.index
        c, o, v = p.Close.values, p.Open.values, p.Volume.values
        sp = spy.reindex(idx).ffill().values
        lr = np.log(c[1:] / c[:-1])
        g = g.reset_index(drop=True)
        past = []  # history of (surp, beat, day0 abn) for this stock
        for _, r in g.iterrows():
            h = r.ts.hour
            if h <= 9: amc = 0
            elif h >= 16: amc = 1
            else: continue
            day = r.ts.tz_localize(None).normalize()
            pos = idx.searchsorted(day + pd.Timedelta(days=amc))
            if pos < 260 or pos >= len(idx): continue
            # guard: reaction session must be within 5 calendar days of the report
            if (idx[pos] - day).days > 5: continue
            pre = pos - 1
            def abn(a, b):  # SPY-relative return from close a to close b
                return (c[b] / c[a] - 1) - (sp[b] / sp[a] - 1)
            f = dict(sym=s, sector=sec.get(s, "NA"), rdate=idx[pos], amc=amc,
                     est=r.est, act=r.act, surp=r.surp)
            # ---- pre-report features
            f["ret5"] = abn(pre - 5, pre); f["ret20"] = abn(pre - 20, pre)
            f["ret60"] = abn(pre - 60, pre); f["ret250"] = abn(pre - 250, pre)
            f["vol20"] = lr[pre - 20:pre].std(); f["vol250"] = lr[pre - 250:pre].std()
            f["dist_hi"] = c[pre] / c[pre - 250:pre + 1].max() - 1
            f["volu_pre"] = v[pre - 5:pre + 1].mean() / max(v[pre - 60:pre - 5].mean(), 1)
            ps = [x for x in past if not np.isnan(x[0])]
            f["n_hist"] = len(past)
            f["beat_rate8"] = np.mean([x[1] for x in past[-8:]]) if past else np.nan
            f["last_surp"] = ps[-1][0] if ps else np.nan
            f["med_surp4"] = np.median([x[0] for x in ps[-4:]]) if ps else np.nan
            f["last_day0"] = past[-1][2] if past else np.nan
            f["avg_day0"] = np.mean([x[2] for x in past[-8:]]) if past else np.nan
            f["avg_absday0"] = np.mean([abs(x[2]) for x in past[-8:]]) if past else np.nan
            # ---- run-up into earnings (announcement premium window): close pre-6 -> close pre
            f["y_runup"] = abn(pre - 6, pre)
            # ---- reaction (known after the report)
            f["gap"] = (o[pos] / c[pre] - 1) - (sp[pos] / sp[pre] - 1) * 0  # raw gap
            f["day0"] = abn(pre, pos)
            f["intraday0"] = c[pos] / o[pos] - 1
            f["volu0"] = v[pos] / max(v[pre - 60:pre - 5].mean(), 1)
            f["day0_z"] = f["day0"] / (f["vol250"] if f["vol250"] > 0 else np.nan)
            # ---- forward drift: enter at NEXT day's open after the reaction day (realistic),
            # exit at close k sessions after reaction. SPY-relative.
            e = pos + 1
            for k in (1, 5, 10, 20, 40, 60):
                x = pos + k
                if x < len(idx) and e < len(idx):
                    spo = sp[pos]  # spy approx entry: prior close (SPY open not stored); small error
                    f[f"post{k}"] = (c[x] / o[e] - 1) - (sp[x] / spo - 1)
                else:
                    f[f"post{k}"] = np.nan
            if not np.isnan(r.act):
                beat = float(r.act > r.est) if not np.isnan(r.est) else 0.5
                past.append((r.surp, beat, f["day0"]))
                out.append(f)
    df = pd.DataFrame(out)
    df["sector"] = df.sector.astype("category")
    df["holdout"] = df.rdate >= HOLDOUT_START
    df.to_pickle(D / "events.pkl")
    return df


if __name__ == "__main__":
    df = build()
    print(len(df), "events;", df.sym.nunique(), "stocks;", df.rdate.min().date(), "->", df.rdate.max().date())
    print("holdout events:", df.holdout.sum())
    print(df.describe().T[["mean", "50%", "std"]].round(4).to_string())
