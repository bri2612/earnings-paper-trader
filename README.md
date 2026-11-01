# Earnings paper trader (S&P 500)

Paper trading only. No real orders are ever placed. See `REPORT.md` for the latest scorecard.

## Daily timing (UK)
- About 10:45pm UK (after the US close): the job runs and logs new signals; a notification is sent.
- Next day, before 2:30pm UK (the US open): that's when the trade would be bought, at the open.
- 20 US trading days later, at the close: the trade would be sold.

## What was learned (Nov 2026 research, ~23,000 earnings reports, 2008–2026)
1. **Predicting the earnings-day direction in advance does not work** with public price and estimate history.
   Every pre-report signal had a correlation of ≤0.03 with the reaction (walk-forward accuracy 50.3%, a coin flip).
2. **Post-earnings drift**: when a stock beats EPS estimates by >5% *and* beats SPY by >3% on report day,
   it tended to outperform for about the next 20 sessions. 2014–Sep 2025: +0.57%/trade over SPY after costs, positive in 12/12 years.
3. **Locked holdout (Oct 2025 onward, tested once, never tuned on): −0.03%/trade over SPY.** The edge has faded.
   With UK retail costs, unhedged, about £200/trade: strategy +£679 vs S&P 500 +£510 over 396 trades, which is not a meaningful difference.
4. Caveats: survivorship bias (only current S&P 500 members), Yahoo data quality, ~52% hit rate even in good years.

## Frozen rule being paper traded
Buy at next open if EPS surprise >5% and day-0 return vs SPY >3%; sell at close 20 sessions later.
Costs are modelled as 0.30% round trip (UK app FX fees). Every trade is compared against SPY over the same dates.

## Go-live test (forward trades only, i.e. logged before entry)
≥100 closed trades, average excess over S&P 500 > +0.5%/trade, worst drawdown better than −20%.

## Files
- `events.py`: builds the event table (all returns relative to SPY, strict pre/post timing)
- `paper.py daily`: updates data, logs signals, scores, writes `REPORT.md`
- `earnings_bot.py`: v1 pre-earnings model (failed, kept for the record) and full data fetch
