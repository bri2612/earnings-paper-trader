# Pre-registered strategies

Rules: a strategy is frozen once added and is never edited. A new idea gets a new letter and counts
only from its start date. History is checked on 2014–Sep 2025 ("old") and the locked year
(Oct 2025 onward) before adding, and the result is recorded here. The go-live decision uses
forward paper trades only (see REPORT.md).

Backtest at the time each was added (UK costs of 0.3% round trip; "beat S&P" = average per trade vs SPY over the same dates; t = t-stat):

| | Strategy | Added | Old: trades / beat S&P / t / years + | Locked year: trades / beat S&P / t |
|---|---|---|---|---|
| A | Beat >5% & up >3% vs S&P on reaction day; hold 20 | 2026-11-01 | 3514 / +0.77% / 6.1 / 12 of 12 | 396 / +0.21% / 0.4 |
| B | Same signal; hold 5 | 2026-11-01 | 3514 / +0.16% / 2.3 / 7 of 12 | 408 / −0.25% / −0.9 |
| C | Same signal; hold 60 | 2026-11-01 | 3514 / +1.10% / 4.2 / 9 of 12 | 375 / −0.07% / −0.1 |
| D | Beat >10% & up >5%; hold 20 | 2026-11-01 | 1541 / +0.99% / 4.6 / 11 of 12 | 182 / +0.73% / 0.8 |
| E | A + volume >3x normal; hold 20 | 2026-11-01 | 1270 / +0.85% / 3.9 / 9 of 12 | 95 / +0.82% / 0.8 |
| F | A + closed above open; hold 20 | 2026-11-01 | 2733 / +0.76% / 5.4 / 12 of 12 | 300 / +0.59% / 0.9 |
| G | A + within 5% of 1-year high; hold 20 | 2026-11-01 | 1214 / +0.58% / 3.3 / 9 of 12 | 124 / +1.47% / 1.4 |
| H | Beat estimates but fell >5% vs S&P (bounce); hold 20 | 2026-11-01 | 1688 / +0.39% / 1.9 / 10 of 12 | 283 / +1.36% / 1.7 |
| I | A + repeat beater (beat in ≥75% of last 8 reports, and the last one); hold 20 | 2026-11-07 (counts reports from 11-06, entry 11-09) | 2077 / +0.51% / 3.4 / 11 of 12 | 293 / +0.34% / 0.5 |

Tested and rejected: predicting the earnings-day direction before the report (walk-forward accuracy 50.3%,
all pre-report signals |correlation| ≤ 0.03).

## Research log (every idea tested, including rejected ones)

Bar for adding: old-period beat S&P > +0.5% with t ≥ 3 and positive in ≥ 9 of 12 years, AND locked year > 0.

### 2026-11-07 (weekly research #1), 3 ideas tested, 1 added
| Idea | Old: trades / beat S&P / t / years + | Locked: trades / beat S&P / t | Decision |
|---|---|---|---|
| A + no run-up (down vs S&P over the prior 20 days) | 1688 / +0.97% / 5.0 / 11 of 12 | 206 / −1.00% / −1.4 | Rejected: negative in locked year |
| A + repeat beater (beat_rate8 ≥ 0.75, last surprise > 0) | 2077 / +0.51% / 3.4 / 11 of 12 | 293 / +0.34% / 0.5 | **Added as I** (only just clears the bar, and overlaps heavily with A) |
| A + calm stock (1y volatility below the old-period median of 1.63%/day) | 1326 / +0.43% / 3.1 / 9 of 12 | 87 / −1.75% / −2.3 | Rejected: below +0.5% and negative in locked year |

Running total of ideas tried after A–H: 3.
