"""
gdxu_report.py
Builds index.html: an interactive weekly GDXU risk page.
Runs automatically on GitHub every weekday (see .github/workflows/update.yml).
Uses the results of the earlier tests: volatility model = HAR + GVZ,
event adjustment from gdxu_v21.py, direction treated as 50/50.
"""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

HORIZON, LEVERAGE, EPS = 5, 3, 1e-10
W_GDX, W_GDXJ = 0.76, 0.24
MIN_TRAIN_YEARS = 3
EVENT_UP, EVENT_QUIET = 1.06, 0.94   # from gdxu_v21.py; update if you rerun it
SCEN_DAYS, SCEN_YEARS = 10, 3
BIG_WEEK_PCT = 0.80                  # "big gold week" = top 20% of weekly gains
LOG_FILE, OUT_FILE = "report_log.csv", "index.html"
FEATURES = ["har_d", "har_w", "har_m", "gvz"]

print("Downloading data...")
raw = yf.download(["GDX", "GDXJ", "GDXU", "^GVZ", "GC=F"], period="max",
                  auto_adjust=True, progress=False, group_by="column")
close = raw["Close"].rename(columns={"^GVZ": "GVZ", "GC=F": "GOLD"})
days = close["GDX"].dropna().index
close = close.ffill(limit=3).loc[days]

rg, rj = close["GDX"].pct_change(), close["GDXJ"].pct_change()
idx = pd.Series(np.where(rj.notna(), W_GDX * rg + W_GDXJ * rj, rg), index=days)
r = np.log1p(idx)
r2 = r ** 2
gdxu = close["GDXU"]
gap = (np.log1p(gdxu.pct_change()) - np.log1p(LEVERAGE * idx)).dropna()
daily_drag = -gap.mean()
g = np.log1p(LEVERAGE * idx) - daily_drag

# ---------------- volatility model and weekly range ----------------
f = pd.DataFrame(index=days)
f["har_d"] = np.log(r2 + EPS)
f["har_w"] = np.log(r2.rolling(5).mean() + EPS)
f["har_m"] = np.log(r2.rolling(22).mean() + EPS)
f["gvz"] = np.log(close["GVZ"])
f["fwd_var"] = r2.rolling(HORIZON).mean().shift(-HORIZON)
f["fwd_g"] = g.rolling(HORIZON).sum().shift(-HORIZON)
cum = g.cumsum()
f["fwd_low"] = pd.concat([cum.shift(-k) - cum for k in range(1, HORIZON + 1)],
                         axis=1).min(axis=1)
f = f.replace([np.inf, -np.inf], np.nan)
hist = f.dropna()

print("Running walk-forward volatility model...")
start = hist.index[0] + pd.DateOffset(years=MIN_TRAIN_YEARS)
z_close, z_low = [], []
for i in np.where(hist.index >= start)[0][::HORIZON]:
    tr = hist.iloc[: i - HORIZON + 1]
    m = LinearRegression().fit(tr[FEATURES], np.log(tr["fwd_var"] + EPS))
    xi = hist.iloc[[i]]
    s = np.sqrt(np.exp(m.predict(xi[FEATURES])[0]) * HORIZON) * LEVERAGE
    z_close.append(xi["fwd_g"].iloc[0] / s)
    z_low.append(xi["fwd_low"].iloc[0] / s)

live = f.dropna(subset=FEATURES).iloc[[-1]]
asof = live.index[-1]
model = LinearRegression().fit(hist[FEATURES], np.log(hist["fwd_var"] + EPS))
sig = np.sqrt(np.exp(model.predict(live[FEATURES])[0]) * HORIZON) * LEVERAGE

bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())
next_days = pd.date_range(asof + bday, periods=HORIZON, freq=bday)
events = []
if os.path.exists("upcoming_events.csv"):
    up = pd.read_csv("upcoming_events.csv", parse_dates=["date"])
    wk_ev = up[up["date"].isin(next_days)]
    events = [f"{e} on {d:%a %d %b}" for d, e in zip(wk_ev["date"], wk_ev["event"])]
sig *= EVENT_UP if events else EVENT_QUIET

price = float(gdxu.loc[:asof].dropna().iloc[-1])
qs = [0.10, 0.25, 0.50, 0.75, 0.90]
close_q = [price * np.exp(np.quantile(z_close, q) * sig) for q in qs]
low_q = [price * np.exp(np.quantile(z_low, q) * sig) for q in (0.10, 0.25, 0.50)]
zq = [float(np.quantile(z_close, q)) for q in qs]

# ---------------- gold scenarios ----------------
gold = close["GOLD"]
rs = gold.index[-1] - pd.DateOffset(years=SCEN_YEARS)
both = pd.concat([np.log(gold).diff().loc[rs:], r.loc[rs:]], axis=1,
                 keys=["gold", "idx"]).dropna()
roll = both.rolling(SCEN_DAYS).sum().dropna().iloc[::SCEN_DAYS]
beta, alpha = np.polyfit(roll["gold"], roll["idx"], 1)
resid = roll["idx"] - (alpha + beta * roll["gold"])
r_sq = 1 - resid.var() / roll["idx"].var()

# ---------------- big gold week ----------------
wk_gold = np.log(gold.resample("W-FRI").last()).diff().dropna()
big_thr = float(wk_gold.quantile(BIG_WEEK_PCT))
this_week = float(wk_gold.iloc[-1])
big_week = this_week >= big_thr
week_rank = float((wk_gold < this_week).mean())

# ===== END OF PART 1 =====
# ---------------- forecast log ----------------
iso = asof.isocalendar()
week_key = f"{iso[0]}-W{iso[1]:02d}"      # one log entry per week; later runs replace it
row = {"week": week_key, "asof": str(asof.date()), "price": round(price, 2),
       "close_p10": round(close_q[0], 2), "close_p90": round(close_q[4], 2),
       "low_p10": round(low_q[0], 2), "big_gold_week": bool(big_week)}
log = pd.read_csv(LOG_FILE) if os.path.exists(LOG_FILE) else pd.DataFrame()
if len(log):
    log = log[log["week"].astype(str) != week_key]
log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)

gs = gdxu.dropna()
scored = []
for _, x in log.iterrows():
    after = gs[gs.index > pd.Timestamp(x["asof"])]
    rec = {"asof": str(x["asof"]), "price": float(x["price"]),
           "p10": float(x["close_p10"]), "p90": float(x["close_p90"]),
           "low10": float(x["low_p10"]), "big": bool(x["big_gold_week"]),
           "close5": None, "low5": None, "in_range": None, "low_held": None,
           "ret2w": None}
    if len(after) >= HORIZON:
        c5, l5 = float(after.iloc[HORIZON - 1]), float(after.iloc[:HORIZON].min())
        rec.update(close5=c5, low5=l5, in_range=bool(rec["p10"] <= c5 <= rec["p90"]),
                   low_held=bool(l5 >= rec["low10"]))
    if len(after) >= 2 * HORIZON:
        rec["ret2w"] = float(after.iloc[2 * HORIZON - 1] / rec["price"] - 1)
    scored.append(rec)
log[["week", "asof", "price", "close_p10", "close_p90", "low_p10", "big_gold_week"]].to_csv(
    LOG_FILE, index=False)

# ---------------- package data for the page ----------------
hist_px = gs.loc[:asof].iloc[-130:]
data = {
    "asof": f"{asof:%A %d %B %Y} (page updated {datetime.now(timezone.utc):%d %b %H:%M} UTC)",
    "weekEnd": f"{next_days[-1]:%A %d %B}",
    "price": price,
    "vol": float(sig),
    "closeQ": [float(v) for v in close_q],
    "lowQ": [float(v) for v in low_q],
    "zq": zq,
    "events": events,
    "history": {"dates": [f"{d:%d %b}" for d in hist_px.index],
                "values": [round(float(v), 2) for v in hist_px.values]},
    "futureDates": [f"{d:%d %b}" for d in next_days],
    "gold": {"now": float(gold.iloc[-1]), "gvz": float(close["GVZ"].dropna().iloc[-1]) / 100,
             "alpha": float(alpha), "beta": float(beta), "r2": float(r_sq),
             "q": [float(v) for v in np.quantile(resid, [0.10, 0.50, 0.90])],
             "decay": float((LEVERAGE ** 2 - LEVERAGE) / 2 * r.iloc[-20:].var() * SCEN_DAYS),
             "drag": float(daily_drag * SCEN_DAYS), "days": SCEN_DAYS},
    "bigWeek": {"flag": bool(big_week), "ret": this_week, "thr": big_thr,
                "rank": week_rank},
    "dragYear": float(1 - np.exp(-daily_drag * 252)),
    "log": scored,
}

# ===== END OF PART 2 =====
