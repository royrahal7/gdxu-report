"""
gdxu_app.py
Builds index.html: an interactive GDXU risk page with a daily and a weekly tab.
Runs automatically on GitHub (see .github/workflows/update.yml): before the open,
hourly while the US market is open, and once after the close.
Before the open, the daily range is re-centred on gold's move since GDXU's last
close (tested in gdxu_preopen.py: narrower range, same 80% coverage).
Weekly model = HAR + GVZ with the tested event adjustment. Daily model = HAR + GVZ
(no event adjustment, as it has not been tested). Direction is treated as 50/50.
"""
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday,
                                    USLaborDay, USMartinLutherKingJr, USMemorialDay,
                                    USPresidentsDay, USThanksgivingDay, nearest_workday)
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

HORIZON, LEVERAGE, EPS = 5, 3, 1e-10
W_GDX, W_GDXJ = 0.76, 0.24
MIN_TRAIN_YEARS = 3
EVENT_UP, EVENT_QUIET = 1.06, 0.94   # weekly only, from gdxu_v21.py
SCEN_DAYS, SCEN_YEARS = 10, 3
BIG_WEEK_PCT = 0.80
LOG_FILE, DAILY_LOG, OUT_FILE = "report_log.csv", "daily_log.csv", "index.html"
FEATURES = ["har_d", "har_w", "har_m", "gvz"]


class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


bday = CustomBusinessDay(calendar=NYSECalendar())

print("Downloading data...")
raw = yf.download(["GDX", "GDXJ", "GDXU", "^GVZ", "GC=F"], period="max",
                  auto_adjust=True, progress=False, group_by="column", threads=False)
close = raw["Close"].rename(columns={"^GVZ": "GVZ", "GC=F": "GOLD"})
gdxu_open_all = raw["Open"]["GDXU"] if "Open" in raw.columns.get_level_values(0) else None
days = close["GDX"].dropna().index
close = close.ffill(limit=3).loc[days]

# During market hours Yahoo includes today's unfinished bar: keep it as the
# live price, but build every forecast from completed closes only.
now_ny = pd.Timestamp.now(tz="America/New_York")
market_open = (now_ny.weekday() < 5 and
               now_ny.time() >= pd.Timestamp("09:30").time() and
               now_ny.time() < pd.Timestamp("16:15").time())
live = None
if market_open and close.index[-1].date() == now_ny.date():
    live = {"price": float(close["GDXU"].iloc[-1]), "time": f"{now_ny:%H:%M}"}
    close = close.iloc[:-1]
days = close.index

rg, rj = close["GDX"].pct_change(), close["GDXJ"].pct_change()
idx = pd.Series(np.where(rj.notna(), W_GDX * rg + W_GDXJ * rj, rg), index=days)
r = np.log1p(idx)
r2 = r ** 2
gdxu = close["GDXU"]
gap = (np.log1p(gdxu.pct_change()) - np.log1p(LEVERAGE * idx)).dropna()
daily_drag = -gap.mean()
g = np.log1p(LEVERAGE * idx) - daily_drag

events_all = pd.DataFrame(columns=["date", "event"])
if os.path.exists("upcoming_events.csv"):
    events_all = pd.read_csv("upcoming_events.csv", parse_dates=["date"])

# ---------------- weekly model ----------------
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

print("Running weekly volatility model...")
start = hist.index[0] + pd.DateOffset(years=MIN_TRAIN_YEARS)
z_close, z_low = [], []
for i in np.where(hist.index >= start)[0][::HORIZON]:
    tr = hist.iloc[: i - HORIZON + 1]
    m = LinearRegression().fit(tr[FEATURES], np.log(tr["fwd_var"] + EPS))
    xi = hist.iloc[[i]]
    s = np.sqrt(np.exp(m.predict(xi[FEATURES])[0]) * HORIZON) * LEVERAGE
    z_close.append(xi["fwd_g"].iloc[0] / s)
    z_low.append(xi["fwd_low"].iloc[0] / s)

live_row = f.dropna(subset=FEATURES).iloc[[-1]]
asof = live_row.index[-1]
model = LinearRegression().fit(hist[FEATURES], np.log(hist["fwd_var"] + EPS))
sig = np.sqrt(np.exp(model.predict(live_row[FEATURES])[0]) * HORIZON) * LEVERAGE

next_days = pd.date_range(asof + bday, periods=HORIZON, freq=bday)
wk_ev = events_all[events_all["date"].isin(next_days)]
events = [f"{e} on {d:%a %d %b}" for d, e in zip(wk_ev["date"], wk_ev["event"])]
sig *= EVENT_UP if events else EVENT_QUIET

price = float(gdxu.loc[:asof].dropna().iloc[-1])
qs = [0.10, 0.25, 0.50, 0.75, 0.90]
close_q = [price * np.exp(np.quantile(z_close, q) * sig) for q in qs]
low_q = [price * np.exp(np.quantile(z_low, q) * sig) for q in (0.10, 0.25, 0.50)]
zq = [float(np.quantile(z_close, q)) for q in qs]

# ---------------- daily model (next trading day) ----------------
print("Running daily volatility model...")
fd = pd.DataFrame(index=days)
fd["har_d"], fd["har_w"], fd["har_m"] = f["har_d"], f["har_w"], f["har_m"]
fd["gvz"] = f["gvz"]
fd["fwd_var"] = r2.shift(-1)
fd["fwd_g"] = g.shift(-1)
fd = fd.replace([np.inf, -np.inf], np.nan)
hd = fd.dropna()
target = np.log(hd["fwd_var"].clip(lower=1e-7))
dstart = np.where(hd.index >= hd.index[0] + pd.DateOffset(years=MIN_TRAIN_YEARS))[0][0]
z_day = []
dm = None
for i in range(dstart, len(hd)):
    if dm is None or (i - dstart) % 5 == 0:
        dm = LinearRegression().fit(hd[FEATURES].iloc[: i], target.iloc[: i])
    s_d = np.sqrt(np.exp(dm.predict(hd[FEATURES].iloc[[i]])[0])) * LEVERAGE
    z_day.append(hd["fwd_g"].iloc[i] / s_d)
z_day = np.array(z_day)
# (empirical z quantiles correct the known low bias of log-variance models)
dm_all = LinearRegression().fit(hd[FEATURES], target)
sig_d = np.sqrt(np.exp(dm_all.predict(live_row[FEATURES])[0])) * LEVERAGE
day_q = [price * np.exp(np.quantile(z_day, q) * sig_d) for q in qs]
target_day = asof + bday
day_ev = events_all[events_all["date"] == target_day]
day_events = [str(e) for e in day_ev["event"]]

# yesterday's (last completed day's) move in context
last_z = float(z_day[-1])
last_move = float(gdxu.loc[:asof].pct_change().iloc[-1])
bigger_than = float((np.abs(z_day[:-1]) < abs(last_z)).mean())

live_info = None
if live is not None:
    live_info = {"price": live["price"], "time": live["time"],
                 "change": live["price"] / price - 1,
                 "inside": bool(day_q[0] <= live["price"] <= day_q[4])}

# ---------------- pre-open adjustment (tested in gdxu_preopen.py) ----------------
NY = "America/New_York"
pre_info = None
if not market_open and gdxu_open_all is not None:
    try:
        gh = yf.download("GC=F", period="730d", interval="60m", progress=False, threads=False)
        gold_h = gh["Close"]
        if isinstance(gold_h, pd.DataFrame):
            gold_h = gold_h.iloc[:, 0]
        gold_h = gold_h.dropna()
        if gold_h.index.tz is None:
            gold_h.index = gold_h.index.tz_localize("UTC")
        gold_h = gold_h.tz_convert(NY).sort_index()

        def gold_at(ts):
            j = gold_h.index.searchsorted(ts, side="right") - 1
            if j < 0 or ts - gold_h.index[j] > pd.Timedelta(hours=3):
                return None
            return float(gold_h.iloc[j])

        gcl = gdxu.dropna()
        gop = gdxu_open_all.reindex(gcl.index)
        hrows = []
        for k in range(1, len(gcl)):
            prev_d, day_d = gcl.index[k - 1], gcl.index[k]
            t1 = pd.Timestamp(prev_d.date()).tz_localize(NY) + pd.Timedelta(hours=15)
            t2 = pd.Timestamp(day_d.date()).tz_localize(NY) + pd.Timedelta(hours=8)
            if t1 < gold_h.index[0] or pd.isna(gop.iloc[k]):
                continue
            g1, g2 = gold_at(t1), gold_at(t2)
            if g1 is None or g2 is None:
                continue
            hrows.append({"gold_on": np.log(g2 / g1),
                          "gap": np.log(gop.iloc[k] / gcl.iloc[k - 1]),
                          "cc": np.log(gcl.iloc[k] / gcl.iloc[k - 1])})
        hp = pd.DataFrame(hrows)
        hp["vol"] = hp["cc"].rolling(20).std().shift(1)
        hp = hp.dropna()
        g_close = gold_at(pd.Timestamp(asof.date()).tz_localize(NY) + pd.Timedelta(hours=15))
        g_now, t_now = float(gold_h.iloc[-1]), gold_h.index[-1]
        t_asof_close = pd.Timestamp(asof.date()).tz_localize(NY) + pd.Timedelta(hours=16)
        if len(hp) >= 100 and g_close is not None and t_now >= t_asof_close:
            b_cc, a_cc = np.polyfit(hp["gold_on"], hp["cc"], 1)
            b_gp, a_gp = np.polyfit(hp["gold_on"], hp["gap"], 1)
            zr = (hp["cc"] - (a_cc + b_cc * hp["gold_on"])) / hp["vol"]
            gr = hp["gap"] - (a_gp + b_gp * hp["gold_on"])
            vol_now = float(np.log(gcl / gcl.shift(1)).iloc[-20:].std())
            g_on = np.log(g_now / g_close)
            day_q = [price * np.exp(a_cc + b_cc * g_on + np.quantile(zr, q) * vol_now) for q in qs]
            o_lo, o_mid, o_hi = (price * np.exp(a_gp + b_gp * g_on + np.quantile(gr, q))
                                 for q in (0.10, 0.50, 0.90))
            pre_info = {"goldMove": float(np.exp(g_on) - 1), "goldNow": g_now,
                        "goldTime": f"{t_now:%a %H:%M}", "open": float(o_mid),
                        "openLo": float(o_lo), "openHi": float(o_hi), "beta": float(b_gp)}
    except Exception as err:
        print("Pre-open step skipped:", err)

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

# ---------------- weekly log ----------------
iso = asof.isocalendar()
week_key = f"{iso[0]}-W{iso[1]:02d}"
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

# ---------------- daily log ----------------
drow = {"target": str(target_day.date()), "base": str(asof.date()), "price": round(price, 2),
        "p10": round(day_q[0], 2), "p25": round(day_q[1], 2),
        "p75": round(day_q[3], 2), "p90": round(day_q[4], 2)}
dlog = pd.read_csv(DAILY_LOG) if os.path.exists(DAILY_LOG) else pd.DataFrame()
already = len(dlog) and (dlog["target"].astype(str) == drow["target"]).any()
if not (market_open and already):          # keep the pre-open forecast once trading starts
    if len(dlog):
        dlog = dlog[dlog["target"].astype(str) != drow["target"]]
    dlog = pd.concat([dlog, pd.DataFrame([drow])], ignore_index=True)
dlog.to_csv(DAILY_LOG, index=False)
dscored = []
for _, x in dlog.iterrows():
    t = pd.Timestamp(x["target"])
    c_t = float(gs.loc[t]) if t in gs.index else None
    dscored.append({"target": f"{t:%a %d %b}", "price": float(x["price"]),
                    "p10": float(x["p10"]), "p90": float(x["p90"]), "close": c_t,
                    "in80": None if c_t is None else bool(x["p10"] <= c_t <= x["p90"]),
                    "in50": None if c_t is None else bool(x["p25"] <= c_t <= x["p75"])})

# ---------------- package data for the page ----------------
hist_px = gs.loc[:asof].iloc[-130:]
data = {
    "asof": f"{asof:%A %d %B %Y}",
    "updated": f"{datetime.now(timezone.utc):%d %b %H:%M} UTC",
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
    "day": {"target": f"{target_day:%A %d %B}", "q": [float(v) for v in day_q],
            "vol": float(sig_d), "events": day_events, "live": live_info, "pre": pre_info,
            "lastDate": f"{asof:%A %d %B}", "lastMove": last_move,
            "biggerThan": bigger_than, "log": dscored},
}

# ---------------- page template ----------------
PAGE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDXU risk tracker</title>
<style>
:root{--bg:#EDEFEF;--ink:#1D2733;--muted:#5E6B78;--gold:#A97C14;--gold2:rgba(169,124,20,.16);
--gold3:rgba(169,124,20,.34);--down:#A83A32;--up:#2E6E6A;--rule:#C9CFD4;--field:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#12171D;--ink:#E3E7EB;--muted:#9AA5B1;--gold:#D7A93B;
--gold2:rgba(215,169,59,.14);--gold3:rgba(215,169,59,.32);--down:#E07A6E;--up:#6FB3AD;--rule:#2A333D;--field:#1A2129}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,"Segoe UI",sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:880px;margin:0 auto;padding:40px 20px 80px}
h1,h2,.big{font-family:Charter,"Sitka Text",Cambria,Georgia,serif;font-weight:600}
h1{font-size:1.1rem;margin:0;color:var(--muted);font-weight:500}
.big{font-size:3.4rem;line-height:1.05;margin:6px 0 4px}
h2{font-size:1.45rem;margin:0 0 6px}
section{padding:36px 0;border-top:1px solid var(--rule)}
header{padding-bottom:0}
.tabs{display:flex;gap:4px;margin-top:22px;border-bottom:1px solid var(--rule)}
.tabs button{font:inherit;background:none;border:0;border-bottom:3px solid transparent;
padding:10px 16px;color:var(--muted);cursor:pointer;margin-bottom:-1px}
.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--gold);font-weight:600}
.tabs button:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
section.first{border-top:0}
[hidden]{display:none!important}
p{margin:.4em 0;max-width:68ch}
.muted{color:var(--muted)}
.flag{display:inline-block;margin-top:10px;padding:6px 12px;border-left:3px solid var(--gold);
background:var(--gold2)}
svg{width:100%;height:auto;display:block;overflow:visible}
svg text{fill:var(--muted);font-size:12px;font-family:system-ui,sans-serif}
svg .lbl{fill:var(--ink);font-size:13px;font-weight:600}
.row{display:flex;gap:24px;flex-wrap:wrap;margin:14px 0}
label{display:block;font-size:.9rem;color:var(--muted);margin-bottom:4px}
input[type=number]{font:inherit;padding:8px 10px;width:200px;background:var(--field);color:var(--ink);
border:1px solid var(--rule);border-radius:4px}
input[type=range]{width:100%;accent-color:var(--gold)}
input:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.out{font-size:1.1rem;margin-top:10px}
.out b{font-family:Charter,"Sitka Text",Cambria,Georgia,serif;font-size:1.25rem}
.dn{color:var(--down)}.upc{color:var(--up)}
table{border-collapse:collapse;width:100%;font-size:.92rem}
.tbl{overflow-x:auto}
th,td{text-align:right;padding:7px 8px;border-bottom:1px solid var(--rule);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500}
</style></head><body><main>
<header>
<h1>GDXU risk tracker, updated <span id="updated"></span></h1>
<div class="big" id="price"></div>
<p id="priceNote"></p>
<div class="tabs" role="tablist">
<button role="tab" id="tbDay" aria-selected="true" aria-controls="tabDay">Today</button>
<button role="tab" id="tbWeek" aria-selected="false" aria-controls="tabWeek">This week</button>
</div>
</header>

<div id="tabDay" role="tabpanel" aria-labelledby="tbDay">
<section class="first">
<h2>Where GDXU could close on <span id="dayTarget"></span></h2>
<div id="preNote"></div>
<p class="muted" id="daySummary"></p>
<svg id="dayRuler" viewBox="0 0 760 150" role="img" aria-label="Range of likely closing prices for the next session"></svg>
<p id="liveNote"></p>
</section>

<section>
<h2>The last session in context</h2>
<p id="lastMove"></p>
</section>

<section>
<h2>Daily track record</h2>
<p class="muted" id="dayLogSummary"></p>
<div class="tbl"><table id="dayLogTable"></table></div>
</section>
</div>

<div id="tabWeek" role="tabpanel" aria-labelledby="tbWeek" hidden>
<section class="first">
<h2>Where GDXU could close by <span id="weekEnd"></span></h2>
<p id="summary"></p>
<div id="flag"></div>
<p class="muted">The dark band holds half of likely outcomes, the light band eight in ten.
The red line is how low it could dip during a rough week, one week in ten.</p>
<svg id="ruler" viewBox="0 0 760 150" role="img" aria-label="Range of likely closing prices"></svg>
</section>

<section>
<h2>The last six months and the week ahead</h2>
<svg id="chart" viewBox="0 0 760 300" role="img" aria-label="GDXU price history with forecast band"></svg>
</section>

<section>
<h2>If gold moves</h2>
<p class="muted" id="goldIntro"></p>
<label for="goldSlider">Gold price in two weeks: <b id="goldVal"></b></label>
<input type="range" id="goldSlider">
<div class="out" id="goldOut"></div>
</section>

<section>
<h2>Size a position</h2>
<div class="row">
<div><label for="pos">Position value ($)</label><input type="number" id="pos" value="10000" min="0" step="500"></div>
<div><label for="maxLoss">Largest weekly loss you would accept ($)</label>
<input type="number" id="maxLoss" value="1000" min="0" step="100"></div>
</div>
<div class="out" id="sizeOut"></div>
</section>

<section>
<h2>Track record</h2>
<p class="muted" id="logSummary"></p>
<div class="tbl"><table id="logTable"></table></div>
</section>

</div>

<section>
<h2>What this can and cannot tell you</h2>
<p>It forecasts how far GDXU is likely to move, today and this week, not which way. Three direction models and
eleven gold price-level signals were tested on up to twenty years of data, and none beat chance.</p>
<p id="dragNote"></p>
<p>Each range will be wrong about one time in five, and rare events such as an early
redemption of the note sit outside any model.</p>
</section>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const usd = v => "$" + v.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pc = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%";
const svgNS = "http://www.w3.org/2000/svg";
function el(svg, tag, attrs, text) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  svg.appendChild(e); return e;
}
function erf(x) {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const ncdf = x => 0.5 * (1 + erf(x / Math.SQRT2));

$("updated").textContent = D.updated;
const L = D.day.live;
$("price").textContent = usd(L ? L.price : D.price);
$("priceNote").textContent = L
  ? "Live at " + L.time + " New York time (Yahoo prices can lag about 15 minutes), " + pc(L.change) +
    " since the last close of " + usd(D.price) + "."
  : "Last close, " + D.asof + "." + (D.day.pre ? " Gold has moved " + pc(D.day.pre.goldMove) +
    " since then; see the likely open below." : "");
function showTab(which) {
  const day = which === "day";
  $("tabDay").hidden = !day; $("tabWeek").hidden = day;
  $("tbDay").setAttribute("aria-selected", day); $("tbWeek").setAttribute("aria-selected", !day);
}
$("tbDay").addEventListener("click", () => showTab("day"));
$("tbWeek").addEventListener("click", () => showTab("week"));
$("weekEnd").textContent = D.weekEnd;
const ev = D.events.length ? "Scheduled this week: " + D.events.join(", ") + "." : "No major scheduled releases this week.";
$("summary").textContent = "From the last close of " + usd(D.price) + ". Expected size of a typical 5-day move: about " + (D.vol * 100).toFixed(0) +
  "% either way. Direction is treated as a coin flip. " + ev;
if (D.bigWeek.flag) {
  $("flag").innerHTML = '<div class="flag">Gold just had one of its strongest weeks (' + pc(D.bigWeek.ret) +
    ', better than ' + (D.bigWeek.rank * 100).toFixed(0) + '% of weeks since 2006). Historically GDXU has tended ' +
    'to give back ground after such weeks. This is an untested pattern being tracked below, not a signal.</div>';
}

// ---- rulers ----
function drawRuler(s, q, lowQ, mark) {
  const live = mark ? mark.v : null;
  const pts = [q[0], q[4], D.price].concat(lowQ ? [lowQ[0]] : []).concat(live ? [live] : []);
  const lo = Math.min(...pts) * 0.97, hi = Math.max(...pts) * 1.03;
  const X = v => 20 + (v - lo) / (hi - lo) * 720;
  const step = [0.5, 1, 2, 5, 10, 20, 25, 50].find(st => (hi - lo) / st <= 10) || 100;
  el(s, "line", {x1: 20, x2: 740, y1: 110, y2: 110, stroke: "var(--rule)"});
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    el(s, "line", {x1: X(t), x2: X(t), y1: 106, y2: 114, stroke: "var(--muted)"});
    el(s, "text", {x: X(t), y: 132, "text-anchor": "middle"}, "$" + (+t.toFixed(1)));
  }
  el(s, "rect", {x: X(q[0]), y: 52, width: X(q[4]) - X(q[0]), height: 46, fill: "var(--gold2)"});
  el(s, "rect", {x: X(q[1]), y: 52, width: X(q[3]) - X(q[1]), height: 46, fill: "var(--gold3)"});
  el(s, "line", {x1: X(q[2]), x2: X(q[2]), y1: 48, y2: 102, stroke: "var(--gold)", "stroke-width": 2});
  if (lowQ) {
    el(s, "line", {x1: X(lowQ[0]), x2: X(lowQ[0]), y1: 40, y2: 110, stroke: "var(--down)",
      "stroke-width": 2, "stroke-dasharray": "4 3"});
    el(s, "text", {x: X(lowQ[0]), y: 32, "text-anchor": "middle", class: "lbl", fill: "var(--down)"}, usd(lowQ[0]));
  }
  if (live) {
    el(s, "line", {x1: X(live), x2: X(live), y1: 40, y2: 110, stroke: "var(--up)",
      "stroke-width": 3});
    el(s, "text", {x: X(live), y: 32, "text-anchor": "middle", class: "lbl", fill: "var(--up)"}, mark.label + " " + usd(live));
  }
  el(s, "line", {x1: X(D.price), x2: X(D.price), y1: 44, y2: 110, stroke: "var(--ink)", "stroke-width": 2});
  el(s, "text", {x: X(D.price), y: 16, "text-anchor": "middle", class: "lbl"}, "Last close " + usd(D.price));
  el(s, "text", {x: X(q[0]), y: 148, "text-anchor": "start", class: "lbl"}, usd(q[0]));
  el(s, "text", {x: X(q[4]), y: 148, "text-anchor": "end", class: "lbl"}, usd(q[4]));
}
drawRuler($("ruler"), D.closeQ, D.lowQ, null);
const P = D.day.pre;
drawRuler($("dayRuler"), D.day.q, null,
  L ? {v: L.price, label: "Live"} : (P ? {v: P.open, label: "Likely open"} : null));

// ---- today tab ----
(function () {
  const d = D.day;
  $("dayTarget").textContent = d.target;
  const ev = d.events.length ? " Scheduled that day: " + d.events.join(", ") + "." : "";
  $("daySummary").textContent = (P ? "Range re-centred on gold's move since GDXU's last close. " :
    "Expected size of a typical one-day move: about " + (d.vol * 100).toFixed(1) + "% either way. ") +
    "The dark band holds half of likely closes, the light band eight in ten." + ev;
  if (P) {
    $("preNote").innerHTML = '<div class="flag">Before the open: gold is <b>' + pc(P.goldMove) +
      "</b> since GDXU's last close (" + usd(P.goldNow) + ", " + P.goldTime + " New York time). " +
      "GDXU is likely to open around <b>" + usd(P.open) + "</b>, with eight in ten opens between " +
      usd(P.openLo) + " and " + usd(P.openHi) + ". This is priced in at the bell and says nothing " +
      "about the rest of the day.</div>";
  }
  if (L) {
    $("liveNote").innerHTML = L.inside
      ? "So far today GDXU is trading <b>inside</b> the eight-in-ten range."
      : "So far today GDXU is trading <b class='dn'>outside</b> the eight-in-ten range, an unusually large move.";
  }
  const pctile = (d.biggerThan * 100).toFixed(0);
  const oneIn = d.biggerThan < 0.999 ? Math.round(1 / (1 - d.biggerThan)) : 1000;
  $("lastMove").innerHTML = "On " + d.lastDate + " GDXU moved <b class='" + (d.lastMove >= 0 ? "upc" : "dn") + "'>" +
    pc(d.lastMove) + "</b>. Relative to what the model expected for that day, the move was bigger than " +
    pctile + "% of days, " + (d.biggerThan >= 0.5 ? "roughly a one-in-" + oneIn + "-day move." : "an ordinary day for GDXU.");
  const done = d.log.filter(r => r.close !== null);
  if (!done.length) {
    $("dayLogSummary").textContent = "No daily forecasts scored yet. Each is scored after the day it covers closes.";
  } else {
    const in80 = done.filter(r => r.in80).length, in50 = done.filter(r => r.in50).length;
    $("dayLogSummary").textContent = "Scored days: " + done.length + ". Closed inside the eight-in-ten range: " + in80 +
      " (" + (in80 / done.length * 100).toFixed(0) + "%, target 80%). Inside the half-of-outcomes band: " + in50 +
      " (" + (in50 / done.length * 100).toFixed(0) + "%, target 50%).";
  }
  const head = "<tr><th>Day</th><th>Prior close</th><th>8-in-10 range</th><th>Actual close</th></tr>";
  const rows = d.log.slice().reverse().slice(0, 20).map(r => "<tr><td>" + r.target + "</td><td>" + usd(r.price) +
    "</td><td>" + usd(r.p10) + " to " + usd(r.p90) + "</td><td>" +
    (r.close === null ? "pending" : "<span class='" + (r.in80 ? "upc" : "dn") + "'>" + usd(r.close) + "</span>") + "</td></tr>");
  $("dayLogTable").innerHTML = head + rows.join("");
})();

// ---- chart ----
(function () {
  const s = $("chart"), h = D.history.values, n = h.length, k = D.futureDates.length;
  const band = [];
  for (let i = 1; i <= k; i++) band.push(D.zq.map(z => D.price * Math.exp(z * D.vol * Math.sqrt(i / k))));
  const all = h.concat(band.flat());
  const lo = Math.min(...all) * 0.97, hi = Math.max(...all) * 1.03;
  const X = i => i < n ? 50 + i / (n - 1) * 560 : 610 + (i - n + 1) / k * 130;
  const Y = v => 270 - (v - lo) / (hi - lo) * 250;
  for (let j = 0; j <= 4; j++) {
    const v = lo + (hi - lo) * j / 4;
    el(s, "line", {x1: 50, x2: 740, y1: Y(v), y2: Y(v), stroke: "var(--rule)"});
    el(s, "text", {x: 44, y: Y(v) + 4, "text-anchor": "end"}, "$" + v.toFixed(0));
  }
  for (let i = 0; i < n - 15; i += 25) el(s, "text", {x: X(i), y: 292, "text-anchor": "middle"}, D.history.dates[i]);
  el(s, "text", {x: X(n - 1), y: 292, "text-anchor": "middle"}, D.history.dates[n - 1]);
  el(s, "line", {x1: X(n - 1), x2: X(n - 1), y1: 20, y2: 270, stroke: "var(--rule)", "stroke-dasharray": "3 3"});
  el(s, "text", {x: X(n + k - 1), y: 292, "text-anchor": "end"}, D.futureDates[k - 1]);
  const poly = (a, b) => {
    const top = [[X(n - 1), Y(D.price)]].concat(band.map((r, i) => [X(n + i), Y(r[b])]));
    const bot = band.map((r, i) => [X(n + i), Y(r[a])]).reverse().concat([[X(n - 1), Y(D.price)]]);
    return top.concat(bot).map(p => p.join(",")).join(" ");
  };
  el(s, "polygon", {points: poly(0, 4), fill: "var(--gold2)"});
  el(s, "polygon", {points: poly(1, 3), fill: "var(--gold3)"});
  el(s, "polyline", {points: h.map((v, i) => X(i) + "," + Y(v)).join(" "), fill: "none",
    stroke: "var(--ink)", "stroke-width": 1.8});
})();

// ---- gold scenarios ----
(function () {
  const G = D.gold, sl = $("goldSlider");
  sl.min = Math.round(G.now * 0.9 / 10) * 10; sl.max = Math.round(G.now * 1.1 / 10) * 10;
  sl.step = 10; sl.value = Math.round(G.now / 10) * 10;
  $("goldIntro").textContent = "Gold is at " + usd(G.now) + ". Over the last three years the miners moved about " +
    G.beta.toFixed(1) + " times as much as gold, and gold explained " + (G.r2 * 100).toFixed(0) +
    "% of their two-week moves. Drag the slider to see what a given gold price would mean for GDXU.";
  const sd = G.gvz * Math.sqrt(G.days / 252);
  function update() {
    const gp = +sl.value, x = Math.log(gp / G.now), base = G.alpha + G.beta * x;
    const [lo, mid, hi] = G.q.map(q => D.price * Math.exp(3 * (base + q) - G.decay - G.drag));
    const chance = 1 - ncdf((x + sd * sd / 2) / sd);
    $("goldVal").textContent = usd(gp) + " (" + pc(gp / G.now - 1) + ")";
    $("goldOut").innerHTML = "GDXU would most likely be around <b>" + usd(mid) + "</b> (" + pc(mid / D.price - 1) +
      "), with eight in ten outcomes between " + usd(lo) + " and " + usd(hi) +
      ".<br><span class='muted'>The options market puts the chance of gold ending at or above this level at about " +
      (chance * 100).toFixed(0) + "%.</span>";
  }
  sl.addEventListener("input", update); update();
})();

// ---- position sizing ----
(function () {
  const bad = D.closeQ[0] / D.price - 1, dip = D.lowQ[0] / D.price - 1;
  function update() {
    const pos = +$("pos").value || 0, ml = +$("maxLoss").value || 0;
    const fit = ml / Math.abs(bad);
    $("sizeOut").innerHTML = "In a bad week (one in ten), " + usd(pos) + " could close the week down about <b class='dn'>" +
      usd(pos * Math.abs(bad)) + "</b> (" + pc(bad) + "), and could dip as far as " + usd(pos * Math.abs(dip)) +
      " below its value along the way.<br>To keep a one-in-ten week under " + usd(ml) +
      ", the position would need to be about <b>" + usd(fit) + "</b> or less at this week's volatility.";
  }
  ["pos", "maxLoss"].forEach(id => $(id).addEventListener("input", update)); update();
})();

// ---- track record ----
(function () {
  const done = D.log.filter(r => r.close5 !== null);
  if (!done.length) {
    $("logSummary").textContent = "No forecasts scored yet. Each weekly run is logged and scored once five trading days have passed.";
  } else {
    const inR = done.filter(r => r.in_range).length, held = done.filter(r => r.low_held).length;
    let txt = "Scored weeks: " + done.length + ". Closed inside the eight-in-ten band: " + inR + " (" +
      (inR / done.length * 100).toFixed(0) + "%, target 80%). Dip stayed above the red line: " + held + " (" +
      (held / done.length * 100).toFixed(0) + "%, target 90%).";
    const two = D.log.filter(r => r.ret2w !== null), bigs = two.filter(r => r.big), rest = two.filter(r => !r.big);
    if (bigs.length && rest.length) {
      const avg = a => a.reduce((s, r) => s + r.ret2w, 0) / a.length;
      txt += " After big gold weeks GDXU averaged " + pc(avg(bigs)) + " over two weeks (" + bigs.length +
        " cases) versus " + pc(avg(rest)) + " otherwise.";
    }
    $("logSummary").textContent = txt;
  }
  const head = "<tr><th>Week of</th><th>Price</th><th>8-in-10 band</th><th>Close 5 days later</th><th>Dip floor</th><th>Lowest close</th></tr>";
  const rows = D.log.slice().reverse().slice(0, 12).map(r => "<tr><td>" + r.asof + (r.big ? " *" : "") + "</td><td>" +
    usd(r.price) + "</td><td>" + usd(r.p10) + " to " + usd(r.p90) + "</td><td>" +
    (r.close5 === null ? "pending" : "<span class='" + (r.in_range ? "upc" : "dn") + "'>" + usd(r.close5) + "</span>") +
    "</td><td>" + usd(r.low10) + "</td><td>" +
    (r.low5 === null ? "pending" : "<span class='" + (r.low_held ? "upc" : "dn") + "'>" + usd(r.low5) + "</span>") +
    "</td></tr>");
  $("logTable").innerHTML = head + rows.join("");
})();
$("dragNote").textContent = "Holding GDXU has cost about " + (D.dragYear * 100).toFixed(0) +
  "% a year compared with three times the index, on top of the losses leverage suffers when prices swing. " +
  "It is built for short holds.";
</script></body></html>'''

with open(OUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(PAGE.replace("__DATA__", json.dumps(data)))
print(f"Saved {OUT_FILE} and {LOG_FILE}. Open {OUT_FILE} in your browser.")

# ===== END OF gdxu_app.py =====
