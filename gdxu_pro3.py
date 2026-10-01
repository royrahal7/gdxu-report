"""
gdxu_pro3.py
Builds index.html (from site_template3.html): the GDXU Pro website, with gold price levels,
a plain-English briefing, chart captions and key-level (confluence) highlighting.
Runs on GitHub Actions: pre-open, every 15 minutes while the US market is open,
and after the close. Everything shown is labelled by evidence:
  tested (passed a pre-registered test), experiment (being scored live), context.
Optional phone alerts: set a repository secret NTFY_TOPIC (see the instructions).
"""
import io
import json
import urllib.request
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
LIVE_START = "2026-09-28 09:30"      # breakout experiment: calls from this time (New York) are live
BRK_STEP, BRK_HOURS = 50.0, 24


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
raw = yf.download(["GDX", "GDXJ", "GDXU", "^GVZ", "GC=F", "DX-Y.NYB", "GLD"], period="max",
                  auto_adjust=True, progress=False, group_by="column", threads=False)
close = raw["Close"].rename(columns={"^GVZ": "GVZ", "GC=F": "GOLD", "DX-Y.NYB": "USD"})
gdxu_hi_all = raw["High"]["GDXU"] if "High" in raw.columns.get_level_values(0) else None
gdxu_lo_all = raw["Low"]["GDXU"] if "Low" in raw.columns.get_level_values(0) else None
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

# ---------------- hourly gold (used by pre-open and breakout watch) ----------------
NY = "America/New_York"
gold_hl = None
try:
    gh = yf.download("GC=F", period="730d", interval="60m", progress=False, threads=False)
    parts = {}
    for fld in ["High", "Low", "Close", "Volume"]:
        if fld not in gh.columns.get_level_values(0):
            continue
        col = gh[fld]
        parts[fld] = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
    gold_hl = pd.DataFrame(parts).dropna(subset=["High", "Low", "Close"])
    if gold_hl.index.tz is None:
        gold_hl.index = gold_hl.index.tz_localize("UTC")
    gold_hl = gold_hl.tz_convert(NY).sort_index()
except Exception as err:
    print("Hourly gold download failed:", err)

# ---------------- pre-open adjustment (tested in gdxu_preopen.py) ----------------
pre_info = None
if not market_open and gdxu_open_all is not None and gold_hl is not None:
    try:
        gold_h = gold_hl["Close"]

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

# ---------------- breakout watch (live experiment) ----------------
brk = None
if gold_hl is not None and len(gold_hl) > 500:
    try:
        bH, bL, bC = (gold_hl[c].to_numpy() for c in ["High", "Low", "Close"])
        bt_idx = gold_hl.index
        nb = len(bC)
        bday_ = pd.Series(bt_idx.date, index=bt_idx)
        dmax = gold_hl.groupby(bday_.values)["High"].max()
        prev_hi = bday_.map(dmax.shift(1)).to_numpy()
        fwd24 = np.full(nb, np.nan)
        fwd24[:nb - BRK_HOURS] = np.log(bC[BRK_HOURS:] / bC[:nb - BRK_HOURS])
        drift24 = float(np.nanmean(fwd24))
        mult = LEVERAGE * beta                     # GDXU move per 1% gold move (short horizon)

        def chain(start):
            calls, t = [], max(start, 1)
            while t < nb:
                lvl = prev_hi[t]
                if np.isnan(lvl) or not (bC[t - 1] <= lvl < bC[t]):
                    t += 1
                    continue
                tgt, stp = lvl + BRK_STEP, lvl - BRK_STEP
                status, end = "open", min(t + BRK_HOURS, nb - 1)
                for u in range(t + 1, end + 1):
                    if bH[u] >= tgt and bL[u] <= stp:
                        status = "tie"; break
                    if bH[u] >= tgt:
                        status = "target"; break
                    if bL[u] <= stp:
                        status = "invalidated"; break
                done24 = t + BRK_HOURS < nb
                if status == "open" and done24:
                    status = "expired"
                calls.append({"time": bt_idx[t], "level": float(lvl), "entry": float(bC[t]),
                              "target": float(tgt), "stop": float(stp), "status": status,
                              "move": float(fwd24[t]) if done24 else None})
                if not done24:
                    break
                t = (u + 1) if status in ("target", "invalidated", "tie") else t + BRK_HOURS + 1
            return calls

        start_i = int(bt_idx.searchsorted(pd.Timestamp(LIVE_START).tz_localize(NY)))
        hist_calls = [c for c in chain(1) if c["time"] < bt_idx[min(start_i, nb - 1)]]
        live_calls = chain(start_i) if start_i < nb else []
        hm = [c["move"] for c in hist_calls if c["move"] is not None]
        lm = [c["move"] for c in live_calls if c["move"] is not None]
        g_now = float(bC[-1])
        ref_px = live["price"] if live is not None else price
        to_gdxu = lambda gp: float(ref_px * np.exp(mult * np.log(gp / g_now)))
        cur = live_calls[-1] if live_calls and live_calls[-1]["status"] == "open" else None
        today_hi = float(prev_hi[-1]) if not np.isnan(prev_hi[-1]) else None
        brk = {
            "start": LIVE_START, "drift": drift24, "goldNow": g_now,
            "goldTime": f"{bt_idx[-1]:%a %H:%M}",
            "histN": len(hm), "histMove": float(np.mean(hm)) if hm else None,
            "watchLevel": today_hi,
            "watchTarget": today_hi + BRK_STEP if today_hi else None,
            "watchStop": today_hi - BRK_STEP if today_hi else None,
            "watchGdxuT": to_gdxu(today_hi + BRK_STEP) if today_hi else None,
            "watchGdxuS": to_gdxu(today_hi - BRK_STEP) if today_hi else None,
            "open": None if cur is None else {
                "time": f"{cur['time']:%a %d %b %H:%M}", "level": cur["level"],
                "entry": cur["entry"], "target": cur["target"], "stop": cur["stop"],
                "gdxuT": to_gdxu(cur["target"]), "gdxuS": to_gdxu(cur["stop"]),
                "expires": f"{cur['time'] + pd.Timedelta(hours=BRK_HOURS):%a %d %b %H:%M}"},
            "calls": [{"time": f"{c['time']:%a %d %b %H:%M}", "level": c["level"],
                       "target": c["target"], "stop": c["stop"], "status": c["status"],
                       "move": c["move"]} for c in live_calls][-15:],
            "liveN": len(lm), "liveMove": float(np.mean(lm)) if lm else None,
            "liveTargets": sum(c["status"] == "target" for c in live_calls),
            "liveStops": sum(c["status"] == "invalidated" for c in live_calls),
            "liveExpired": sum(c["status"] == "expired" for c in live_calls),
        }
    except Exception as err:
        print("Breakout watch skipped:", err)

# ---------------- gold price levels (context, scored live against fake levels) ----------------
LVL_LOG = "levels_log.csv"
TOL_TOUCH, REJECT, BREAK = 0.001, 0.005, 0.003
levels, lvl_stats, gold_daily = [], None, []
fib_info, ma_info, gdo = None, {}, None
try:
    gdo = pd.concat([raw[f_]["GC=F"] for f_ in ["Open", "High", "Low", "Close"]], axis=1,
                    keys=["o", "h", "l", "c"]).dropna()
    gold_daily = [[int(pd.Timestamp(t).tz_localize(NY).timestamp()), round(float(v.o), 1), round(float(v.h), 1),
                   round(float(v.l), 1), round(float(v.c), 1)] for t, v in gdo.iloc[-130:].iterrows()]
    g_ref = float(gold_hl["Close"].iloc[-1]) if gold_hl is not None else float(gdo["c"].iloc[-1])
    ref_x = live["price"] if live is not None else price
    to_x = lambda gp: float(ref_x * np.exp(LEVERAGE * beta * np.log(gp / g_ref)))
    cand = []

    def addl(typ, label, p, strength=1.0, lo=None, hi=None):
        if p and np.isfinite(p) and abs(p / g_ref - 1) <= 0.06:
            cand.append({"type": typ, "label": label, "price": float(p), "lo": float(lo or p), "hi": float(hi or p),
                         "strength": float(strength)})

    # swing zones from daily (6 months) and 4-hour (60 days) turning points
    def pivots(h_, l_, k):
        out = []
        for i in range(k, len(h_) - k):
            if h_[i] == h_[i - k:i + k + 1].max():
                out.append((h_[i], i / len(h_)))
            if l_[i] == l_[i - k:i + k + 1].min():
                out.append((l_[i], i / len(h_)))
        return out
    d6 = gdo.iloc[-126:]
    pts = pivots(d6["h"].to_numpy(), d6["l"].to_numpy(), 3)
    if gold_hl is not None:
        h4 = gold_hl.resample("4h").agg({"High": "max", "Low": "min", "Close": "last"}).dropna().iloc[-360:]
        pts += pivots(h4["High"].to_numpy(), h4["Low"].to_numpy(), 4)
    pts.sort()
    zone = []
    for p_, rec in pts + [(np.inf, 0)]:
        if zone and p_ > zone[0][0] * 1.004:
            if len(zone) >= 2:
                ps = [z[0] for z in zone]
                addl("swing", f"Swing zone ({len(zone)} touches)", float(np.mean(ps)),
                     len(zone) + max(z[1] for z in zone), min(ps), max(ps))
            zone = []
        zone.append((p_, rec))

    # recent extremes
    if gold_hl is not None:
        gday = pd.Series(gold_hl.index.date, index=gold_hl.index)
        dmx, dmn = gold_hl.groupby(gday.values)["High"].max(), gold_hl.groupby(gday.values)["Low"].min()
        if len(dmx) >= 2:
            addl("extreme", "Yesterday's high", float(dmx.iloc[-2]), 2)
            addl("extreme", "Yesterday's low", float(dmn.iloc[-2]), 2)
    wk = gdo.resample("W-FRI").agg({"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()
    wk = wk[wk.index < pd.Timestamp(now_ny.date())]
    if len(wk):
        lw = wk.iloc[-1]
        addl("extreme", "Last week's high", float(lw.h), 2)
        addl("extreme", "Last week's low", float(lw.l), 2)
        pv = (lw.h + lw.l + lw.c) / 3
        for lab_, val in [("Pivot P", pv), ("Pivot R1", 2 * pv - lw.l), ("Pivot S1", 2 * pv - lw.h),
                          ("Pivot R2", pv + (lw.h - lw.l)), ("Pivot S2", pv - (lw.h - lw.l))]:
            addl("pivot", lab_, float(val), 1.5)

    # round numbers
    base50 = np.floor(g_ref / 50) * 50
    for p_ in [base50 - 50, base50, base50 + 50, base50 + 100]:
        addl("round", "Round number", p_, 1.5 if p_ % 100 == 0 else 1.0)

    # Fibonacci retracements of the last 6-month swing
    hi_i, lo_i = int(d6["h"].to_numpy().argmax()), int(d6["l"].to_numpy().argmin())
    H_, L_ = float(d6["h"].max()), float(d6["l"].min())
    fib_info = {"high": H_, "low": L_, "down": bool(hi_i < lo_i)}
    for fr_ in [0.236, 0.382, 0.5, 0.618, 0.786]:
        p_ = H_ - fr_ * (H_ - L_) if lo_i < hi_i else L_ + fr_ * (H_ - L_)
        addl("fib", f"Fib {fr_ * 100:.1f}%", p_, 1.5 if fr_ in (0.382, 0.5, 0.618) else 1.0)

    # volume nodes (last 30 days of hourly gold futures volume)
    if gold_hl is not None and "Volume" in gold_hl and gold_hl["Volume"].notna().sum() > 100:
        vp = gold_hl[gold_hl.index >= gold_hl.index[-1] - pd.Timedelta(days=30)].dropna()
        typ_px = (vp["High"] + vp["Low"] + vp["Close"]) / 3
        bin_ = max(5.0, round(g_ref * 0.001))
        prof = vp["Volume"].groupby((typ_px / bin_).round() * bin_).sum().sort_values(ascending=False)
        chosen = []
        for p_, v_ in prof.items():
            if all(abs(p_ / c_ - 1) > 0.01 for c_ in chosen):
                chosen.append(p_)
            if len(chosen) == 4:
                break
        for j_, p_ in enumerate(chosen):
            addl("volume", "Point of control" if j_ == 0 else "Volume node", float(p_), 2 if j_ == 0 else 1)

    # moving averages
    for n_ in (50, 200):
        if len(gdo) > n_:
            ma_info[n_] = float(gdo["c"].iloc[-n_:].mean())
            addl("ma", f"{n_}-day average", ma_info[n_], 1.5)

    for c_ in cand:
        c_["kind"] = "res" if c_["price"] > g_ref else "sup"
        c_["dist"] = c_["price"] / g_ref - 1
        c_["gdxu"] = to_x(c_["price"])
    keep = []
    for typ in set(c_["type"] for c_ in cand):            # at most 3 per type on each side, nearest first
        for kd in ("res", "sup"):
            grp = sorted([c_ for c_ in cand if c_["type"] == typ and c_["kind"] == kd], key=lambda c_: abs(c_["dist"]))
            keep += grp[:3]
    levels = sorted(keep, key=lambda c_: c_["price"], reverse=True)
    for c_ in levels:                                    # key level = another method within 0.3%
        c_["key"] = any(o_["type"] != c_["type"] and abs(o_["price"] / c_["price"] - 1) <= 0.003 for o_ in cand)

    # ---- live scoring: log today's levels (plus fake ones nearby) once, then score reactions ----
    if gold_hl is not None:
        today_key = str(gold_hl.index[-1].date())
        cols_ = ["date", "type", "kind", "price", "fake"]
        lg_ = pd.read_csv(LVL_LOG) if os.path.exists(LVL_LOG) else pd.DataFrame(columns=cols_)
        if today_key not in set(lg_["date"].astype(str)) and pd.Timestamp(today_key) >= pd.Timestamp(LIVE_START[:10]):
            new_ = []
            for k_, c_ in enumerate(levels):
                new_.append([today_key, c_["type"], c_["kind"], round(c_["price"], 2), 0])
                off = 1 + 0.004 * (1 if k_ % 2 else -1)
                new_.append([today_key, c_["type"], c_["kind"], round(c_["price"] * off, 2), 1])
            lg_ = pd.concat([lg_, pd.DataFrame(new_, columns=cols_)], ignore_index=True)
            lg_.to_csv(LVL_LOG, index=False)

        def react(date_, kind_, p_):
            t0 = pd.Timestamp(date_).tz_localize(NY) + pd.Timedelta(hours=9)
            t1 = t0 + pd.Timedelta(hours=20)
            if gold_hl.index[-1] < t1:
                return "pending"
            seg = gold_hl[(gold_hl.index >= t0) & (gold_hl.index < t1)]
            Hs, Ls, Cs = seg["High"].to_numpy(), seg["Low"].to_numpy(), seg["Close"].to_numpy()
            for i_ in range(len(seg)):
                hit = Hs[i_] >= p_ * (1 - TOL_TOUCH) if kind_ == "res" else Ls[i_] <= p_ * (1 + TOL_TOUCH)
                if not hit:
                    continue
                for j_ in range(i_, min(i_ + 7, len(seg))):
                    if kind_ == "res":
                        if Cs[j_] <= p_ * (1 - REJECT):
                            return "reject"
                        if Cs[j_] >= p_ * (1 + BREAK):
                            return "break"
                    else:
                        if Cs[j_] >= p_ * (1 + REJECT):
                            return "reject"
                        if Cs[j_] <= p_ * (1 - BREAK):
                            return "break"
                return "unclear"
            return "untouched"

        if len(lg_):
            lg_["out"] = [react(d_, k_, float(p_)) for d_, k_, p_ in zip(lg_["date"], lg_["kind"], lg_["price"])]
            stats = {}
            for (typ, fake), grp in lg_.groupby(["type", "fake"]):
                stats.setdefault(typ, {})["fake" if fake else "real"] = {
                    "touched": int(grp["out"].isin(["reject", "break", "unclear"]).sum()),
                    "reject": int((grp["out"] == "reject").sum()), "break": int((grp["out"] == "break").sum())}
            lvl_stats = {"start": LIVE_START[:10], "days": int(lg_["date"].nunique()), "types": stats}
except Exception as err:
    print("Levels skipped:", err)

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


# =====================================================================
# PRO SECTIONS
# =====================================================================
def ny_index(s):
    s = s.dropna().copy()
    s.index = s.index.tz_convert(NY) if s.index.tz else s.index.tz_localize("UTC").tz_convert(NY)
    return s.sort_index()


def fld(raw_, f, t):
    s = raw_[f][t] if isinstance(raw_.columns, pd.MultiIndex) else raw_[f]
    return ny_index(s)


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore").replace("\x00", "")


SLOTS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
SLOT_START = [570, 630, 690, 750, 810, 870, 930]
SLOT_LEN = [60, 60, 60, 60, 60, 60, 30]
OPEN_MIN, CLOSE_MIN = 570, 960
ACTIVITY = ["very quiet", "quiet", "normal", "busy", "heavy"]


def hgrid(s):
    df = pd.DataFrame({"v": s.values, "day": s.index.date, "slot": s.index.strftime("%H:%M")})
    return df.pivot_table(index="day", columns="slot", values="v", aggfunc="last").reindex(columns=SLOTS)


def calibrate_intraday():
    rh = yf.download(["GDXU", "GLD"], period="730d", interval="60m", auto_adjust=True,
                     progress=False, threads=False, group_by="column")
    Ox, Cx = hgrid(fld(rh, "Open", "GDXU")), hgrid(fld(rh, "Close", "GDXU"))
    Og, Cg, Vg = hgrid(fld(rh, "Open", "GLD")), hgrid(fld(rh, "Close", "GLD")), hgrid(fld(rh, "Volume", "GLD"))
    ok = Ox.notna().all(1) & Cx.notna().all(1) & Og.notna().all(1) & Cg.notna().all(1) & Vg.notna().all(1)
    Ox, Cx, Og, Cg, Vg = (d_[ok] for d_ in (Ox, Cx, Og, Cg, Vg))
    rx, rgl = np.log(Cx / Ox), np.log(Cg / Og)
    v_tr = (rx ** 2).rolling(20).mean().shift(1).to_numpy()
    rxn, Cxn = rx.to_numpy(), Cx.to_numpy()
    z_rest = []
    for k in range(6):
        z_ = np.log(Cxn[:, 6] / Cxn[:, k]) / np.sqrt(np.nansum(v_tr[:, k + 1:], axis=1))
        z_rest.append(z_[np.isfinite(z_)])
    z_rest = np.concatenate(z_rest)
    A = np.abs(rgl.to_numpy()) / np.abs(rgl).rolling(20).median().shift(1).to_numpy()
    V = Vg.to_numpy() / Vg.rolling(20).median().shift(1).to_numpy()
    S = np.sqrt(A * V)[:, :6]
    Y = rxn[:, 1:] / np.sqrt(v_tr[:, 1:])
    mm = np.isfinite(S) & np.isfinite(Y)
    s_all, y_all = S[mm], Y[mm]
    edges = np.quantile(s_all, [0.2, 0.4, 0.6, 0.8])
    bkt = np.searchsorted(edges, s_all)
    base = np.mean(y_all ** 2)
    mult = [float(np.sqrt(np.mean(y_all[bkt == b] ** 2) / base)) for b in range(5)]
    last = slice(-20, None)
    return {"v_now": np.nanmean(rxn[last] ** 2, axis=0),
            "zq_rest": np.quantile(z_rest, qs), "zq_next": np.quantile(y_all / np.array(mult)[bkt], qs),
            "edges": edges, "mult": mult,
            "med_a": np.nanmedian(np.abs(rgl.to_numpy())[last], axis=0),
            "med_v": np.nanmedian(Vg.to_numpy()[last], axis=0)}


def var_between(cal, m0, m1):
    tot = 0.0
    for j in range(7):
        a, b = SLOT_START[j], SLOT_START[j] + SLOT_LEN[j]
        tot += cal["v_now"][j] * max(0, min(b, m1) - max(a, m0)) / SLOT_LEN[j]
    return tot


# ---------------- intraday candles, rest of day, next-hour risk ----------------
print("Intraday data...")
intraday = {"gdxu": [], "gold": []}
rest_info, risk_info = None, None
try:
    r5 = yf.download(["GDXU", "GC=F", "GLD"], period="5d", interval="5m", auto_adjust=True,
                     progress=False, threads=False, group_by="column")
    x5 = pd.concat([fld(r5, f, "GDXU") for f in ["Open", "High", "Low", "Close"]], axis=1,
                   keys=["o", "h", "l", "c"]).dropna()
    keep_days = sorted(set(x5.index.date))[-2:]
    x5 = x5[np.isin(x5.index.date, keep_days)]
    g5 = pd.concat([fld(r5, f, "GC=F") for f in ["Open", "High", "Low", "Close"]], axis=1,
                   keys=["o", "h", "l", "c"]).dropna()
    g5 = g5[g5.index >= x5.index[0] - pd.Timedelta(hours=2)]
    for key, df5 in [("gdxu", x5), ("gold", g5)]:
        intraday[key] = [[int(t.timestamp()), round(float(r_.o), 2), round(float(r_.h), 2),
                          round(float(r_.l), 2), round(float(r_.c), 2)] for t, r_ in df5.iterrows()]
    if market_open and live is not None:
        cal = calibrate_intraday()
        minute = now_ny.hour * 60 + now_ny.minute
        sig_rest = np.sqrt(var_between(cal, max(minute, OPEN_MIN), CLOSE_MIN))
        rest_info = {"q": [live["price"] * float(np.exp(z_ * sig_rest)) for z_ in cal["zq_rest"]],
                     "sig": float(sig_rest), "minsLeft": int(CLOSE_MIN - minute)}
        if minute >= OPEN_MIN + 60:
            gc5, go5, gv5 = fld(r5, "Close", "GLD"), fld(r5, "Open", "GLD"), fld(r5, "Volume", "GLD")
            since = now_ny - pd.Timedelta(minutes=60)
            wc, wo, wv = gc5[gc5.index >= since], go5[go5.index >= since], gv5[gv5.index >= since]
            k_ = max(0, min(5, (minute - 30 - OPEN_MIN) // 60))
            mv = float(np.log(wc.iloc[-1] / wo.iloc[0]))
            s_ = np.sqrt(abs(mv) / cal["med_a"][k_] * float(wv.sum()) / cal["med_v"][k_])
            b_ = int(np.searchsorted(cal["edges"], s_))
            sig_next = np.sqrt(var_between(cal, minute, min(minute + 60, CLOSE_MIN)))
            risk_info = {"level": ACTIVITY[b_], "bucket": b_, "mult": cal["mult"][b_], "goldMove": mv,
                         "volRatio": float(wv.sum() / cal["med_v"][k_]),
                         "q": [live["price"] * float(np.exp(z_ * cal["mult"][b_] * sig_next))
                               for z_ in cal["zq_next"]]}
except Exception as err:
    print("Intraday section skipped:", err)

# ---------------- gold drivers (dollar, real yields) ----------------
print("Gold drivers...")
drivers_info = None
try:
    real = None
    try:
        t_ = fetch_text("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10")
        fr = pd.read_csv(io.StringIO(t_), na_values=".")
        real = pd.Series(pd.to_numeric(fr.iloc[:, 1], errors="coerce").values,
                         index=pd.to_datetime(fr.iloc[:, 0])).dropna()
    except Exception as err:
        print("  Real yields unavailable:", err)
    dd = pd.DataFrame({"gold": close["GOLD"], "usd": close["USD"]})
    if real is not None:
        dd["real"] = real.reindex(dd.index).ffill(limit=3)
    dd = dd.dropna()
    cols_x = ["usd"] + (["real"] if "real" in dd else [])
    ch = pd.DataFrame({"gold": np.log(dd["gold"]).diff(), "usd": np.log(dd["usd"]).diff()})
    if "real" in dd:
        ch["real"] = dd["real"].diff() * 100
    wk_ = ch.rolling(5).sum().iloc[::-5][::-1].dropna().iloc[-104:]      # last 2 years of weeks
    Xw = np.column_stack([np.ones(len(wk_))] + [wk_[c_] for c_ in cols_x])
    bw, *_ = np.linalg.lstsq(Xw, wk_["gold"], rcond=None)
    r2w = 1 - np.var(wk_["gold"] - Xw @ bw) / np.var(wk_["gold"])
    last5 = ch.iloc[-5:].sum()
    contrib = {c_: float(bw[i + 1] * last5[c_]) for i, c_ in enumerate(cols_x)}
    drivers_info = {"week": float(last5["gold"]), "contrib": contrib,
                    "other": float(last5["gold"] - sum(contrib.values())),
                    "r2": float(r2w), "hasReal": "real" in dd,
                    "usdMove": float(last5["usd"]),
                    "realMove": float(last5["real"]) if "real" in dd else None}
except Exception as err:
    print("Drivers skipped:", err)

# ---------------- gold trend (12 and 6 months) ----------------
trend_info = None
try:
    try:
        gm = close["GOLD"].resample("ME").last()
    except ValueError:
        gm = close["GOLD"].resample("M").last()
    gm = gm.dropna()
    trend_info = {"m12": float(np.log(gm.iloc[-1] / gm.iloc[-13])), "m6": float(np.log(gm.iloc[-1] / gm.iloc[-7]))}
except Exception as err:
    print("Trend skipped:", err)

# ---------------- options pricing (volatility premium) ----------------
vrp_info = None
try:
    vv = pd.DataFrame({"gld": close["GLD"], "gvz": close["GVZ"]}).dropna()
    lr_ = np.log(vv["gld"]).diff()
    vv["fwd"] = lr_[::-1].rolling(21).std()[::-1].shift(-1) * np.sqrt(252) * 100
    vv["rv1"], vv["rv5"], vv["rv22"] = (lr_.abs() * np.sqrt(252), lr_.rolling(5).std() * np.sqrt(252),
                                        lr_.rolling(22).std() * np.sqrt(252))
    fit = vv.dropna()
    Xc = ["rv1", "rv5", "rv22"]
    hm_ = LinearRegression().fit(np.log(fit[Xc] + 1e-6), np.log(fit["fwd"] / 100))
    cur = vv.dropna(subset=Xc).iloc[[-1]]
    ours = float(np.exp(hm_.predict(np.log(cur[Xc] + 1e-6))[0]) * 100)
    samp = fit.iloc[::21]
    prem = samp["gvz"] - samp["fwd"]
    trail = (lr_.rolling(21).std() * np.sqrt(252) * 100).iloc[-260:]
    vrp_info = {"gvz": float(cur["gvz"].iloc[0]), "ours": ours, "typical": float(prem.mean()),
                "share": float((prem > 0).mean()),
                "series": [[int(pd.Timestamp(t).timestamp()), round(float(a), 1), round(float(b), 1)]
                           for t, a, b in zip(trail.index, vv["gvz"].reindex(trail.index).ffill(), trail)
                           if np.isfinite(a) and np.isfinite(b)]}
except Exception as err:
    print("Volatility premium skipped:", err)

# ---------------- hedge-fund positioning (CFTC) ----------------
cot_info = None
try:
    t_ = fetch_text("https://publicreporting.cftc.gov/resource/72hh-3qpy.csv?"
                    "cftc_contract_market_code=088691&$limit=5000")
    cot = pd.read_csv(io.StringIO(t_))
    def fc(*k):
        return next(c_ for c_ in cot.columns if all(x_ in c_.lower() for x_ in k))
    cot["date"] = pd.to_datetime(cot[fc("report_date")].astype(str).str[:10])
    cot = cot.sort_values("date").drop_duplicates("date")
    net = ((pd.to_numeric(cot[fc("m_money_positions_long_all")]) -
            pd.to_numeric(cot[fc("m_money_positions_short_all")])) /
           pd.to_numeric(cot[fc("open_interest_all")]))
    net.index = cot["date"]
    z3 = (net.iloc[-1] - net.iloc[-156:].mean()) / net.iloc[-156:].std()
    cot_info = {"net": float(net.iloc[-1]), "z": float(z3), "date": f"{net.index[-1]:%d %b %Y}",
                "series": [[int(t.timestamp()), round(float(v) * 100, 1)] for t, v in net.iloc[-156:].items()]}
except Exception as err:
    print("Positioning skipped:", err)

# ---------------- signal board ----------------
board = []


def add(name, value, detail, level, evidence):
    board.append({"name": name, "value": value, "detail": detail, "level": level, "evidence": evidence})


rv20 = np.log(gdxu.dropna() / gdxu.dropna().shift(1)).rolling(20).std().dropna()
pct_vol = float((rv20.iloc[-500:] < rv20.iloc[-1]).mean())
add("Volatility regime", ["Calm", "Normal", "Elevated", "Extreme"][int(np.searchsorted([0.25, 0.75, 0.95], pct_vol))],
    f"GDXU's recent swings are bigger than {pct_vol * 100:.0f}% of the last two years. Typical day now: "
    f"about {sig_d * 100:.1f}% either way.", "alert" if pct_vol > 0.95 else "watch" if pct_vol > 0.75 else "ok",
    "tested")
if risk_info:
    add("Next-hour risk", risk_info["level"].capitalize(),
        f"Gold's last hour was {risk_info['level']}; GDXU's next hour likely about {risk_info['mult']:.1f}x normal size.",
        "alert" if risk_info["bucket"] == 4 else "watch" if risk_info["bucket"] == 3 else "ok", "tested")
else:
    add("Next-hour risk", "Off", "Available during market hours after 10:30 New York time.", "info", "tested")
if day_events or events:
    add("Event risk", "Today" if day_events else "This week",
        ("Today: " + ", ".join(day_events) + ". ") if day_events else ("This week: " + ", ".join(events) + ". ")
        + "Expect bigger moves around the release.", "watch", "tested")
else:
    add("Event risk", "None", "No FOMC, CPI or jobs report scheduled this week.", "ok", "tested")
if pre_info:
    add("Pre-open gap", f"{(pre_info['open'] / price - 1) * 100:+.1f}%",
        f"Gold {pre_info['goldMove'] * 100:+.1f}% since GDXU's last close; likely open about ${pre_info['open']:.2f}.",
        "watch" if abs(pre_info['open'] / price - 1) > 0.04 else "ok", "tested")
if trend_info:
    up12 = trend_info["m12"] > 0
    add("Gold 12-month trend", "Up" if up12 else "Down",
        f"Gold {trend_info['m12'] * 100:+.0f}% over 12 months, {trend_info['m6'] * 100:+.0f}% over 6. "
        "A slow backdrop, not a trading signal.", "ok" if up12 else "watch", "tested")
if vrp_info:
    gap_ = vrp_info["gvz"] - vrp_info["ours"]
    add("Options pricing", f"{gap_:+.1f} pts",
        f"Gold options price {vrp_info['gvz']:.1f}% volatility vs our {vrp_info['ours']:.1f}% forecast "
        f"(historically options run about {vrp_info['typical']:+.1f} pts rich).", "info", "tested")
if brk:
    if brk.get("open"):
        add("Breakout watch", "Call open", f"Gold broke yesterday's high; target ${brk['open']['target']:,.0f}.",
            "watch", "experiment")
    else:
        add("Breakout watch", "Watching", f"A close above ${brk['watchLevel']:,.0f} triggers a call."
            if brk.get("watchLevel") else "No level available.", "info", "experiment")
add("Big gold week", "Yes" if big_week else "No",
    "Gold's week was in its top 20%; GDXU has tended to give back ground after such weeks (untested)."
    if big_week else "Nothing unusual about gold's week.", "watch" if big_week else "ok", "experiment")
if cot_info:
    add("Hedge-fund positioning", f"{cot_info['net'] * 100:.0f}% net long",
        f"{cot_info['z']:+.1f} standard deviations vs the past 3 years (as of {cot_info['date']}). "
        "Not predictive in testing; context only.", "watch" if abs(cot_info["z"]) > 1.5 else "ok", "context")
if drivers_info:
    top_ = max(drivers_info["contrib"].items(), key=lambda kv: abs(kv[1]))
    nm_ = {"usd": "the dollar", "real": "real yields"}[top_[0]]
    add("What's driving gold", f"{drivers_info['week'] * 100:+.1f}% this week",
        f"Biggest identified driver: {nm_} ({top_[1] * 100:+.1f} pts); gold's own flows "
        f"{drivers_info['other'] * 100:+.1f} pts.", "info", "context")
var_idx = float(r.iloc[-20:].var())
decay_day = (LEVERAGE ** 2 - LEVERAGE) / 2 * var_idx + daily_drag
add("Holding cost", f"{decay_day * 100:.2f}% a day",
    f"Volatility decay plus measured drag at today's volatility, about {decay_day * 100 * 5:.1f}% a week "
    "if the index goes nowhere.", "watch" if decay_day > 0.004 else "info", "context")

# ---------------- phone alerts (optional) ----------------
alerts_out = []
topic = os.environ.get("NTFY_TOPIC", "").strip()
sent_file = "alerts_sent.json"
sent = json.load(open(sent_file)) if os.path.exists(sent_file) else []
if brk and brk.get("open"):
    alerts_out.append((f"brk-{brk['open']['time']}", "Breakout call (experiment)",
                       f"Gold broke yesterday's high {brk['open']['level']:,.0f}. Target {brk['open']['target']:,.0f} "
                       f"(GDXU ~${brk['open']['gdxuT']:.2f}), invalidated {brk['open']['stop']:,.0f}."))
if risk_info and risk_info["bucket"] == 4:
    alerts_out.append((f"risk-{now_ny:%Y%m%d%H}", "Heavy gold activity",
                       f"GDXU's next hour likely ~{risk_info['mult']:.1f}x normal size. Check stops."))
if live and rest_info and not (day_q[0] <= live["price"] <= day_q[4]):
    alerts_out.append((f"range-{now_ny:%Y%m%d}", "GDXU outside today's 80% range",
                       f"GDXU ${live['price']:.2f} vs range ${day_q[0]:.2f}-${day_q[4]:.2f}."))
for d_, e_ in zip(events_all["date"], events_all["event"]):
    tm = {"FOMC": "14:00"}.get(str(e_).upper(), "08:30")
    when = pd.Timestamp(f"{pd.Timestamp(d_).date()} {tm}", tz=NY)
    if 0 <= (when - now_ny).total_seconds() / 60 <= 75:
        alerts_out.append((f"ev-{e_}-{when:%Y%m%d}", f"{e_} soon", f"{e_} at {tm} New York. Volatility usually jumps."))
if topic:
    for key, title, msg in alerts_out:
        if key in sent:
            continue
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"https://ntfy.sh/{topic}", data=msg.encode(), headers={"Title": title}), timeout=20)
            sent.append(key)
        except Exception as err:
            print("Alert failed:", err)
    json.dump(sent[-300:], open(sent_file, "w"))

# ---------------- plain-English briefing and chart captions ----------------
brief, captions = None, {}
try:
    TYPE_NAME = {"swing": "swing zone", "extreme": "recent high/low", "round": "round number", "pivot": "weekly pivot",
                 "fib": "Fibonacci level", "volume": "volume node", "ma": "moving average"}
    g_now_b = float(gold_hl["Close"].iloc[-1]) if gold_hl is not None else float(gold.iloc[-1])
    sd5 = float(close["GVZ"].dropna().iloc[-1]) / 100 * np.sqrt(5 / 252)
    from scipy.stats import norm as _norm
    touch = lambda p_: int(min(99, round(2 * (1 - _norm.cdf(abs(np.log(p_ / g_now_b)) / sd5)) * 100)))
    ref_b = live["price"] if live is not None else price
    gx_b = lambda p_: float(ref_b * np.exp(LEVERAGE * beta * np.log(p_ / g_now_b)))
    money = lambda v_: f"${v_:,.0f}"
    and_join = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

    # merge levels within 0.3% into "areas" so the briefing talks about places, not lines
    areas = []
    for l_ in sorted(levels, key=lambda x_: x_["price"]):
        if areas and l_["price"] / areas[-1]["hi"] - 1 <= 0.003:
            a_ = areas[-1]
            a_["items"].append(l_)
            a_["hi"] = max(a_["hi"], l_["hi"])
        else:
            areas.append({"items": [l_], "lo": l_["lo"], "hi": l_["hi"]})
    for a_ in areas:
        a_["price"] = float(np.mean([i_["price"] for i_ in a_["items"]]))
        a_["types"] = sorted(set(i_["type"] for i_ in a_["items"]))
        touches = max([int("".join(ch for ch in i_["label"] if ch.isdigit()) or 0)
                       for i_ in a_["items"] if i_["type"] == "swing"] or [0])
        a_["key"] = len(a_["types"]) >= 2 or touches >= 3
        a_["name"] = and_join([
            (f"swing zone ({touches} touches)" if t_ == "swing" else "round number" if t_ == "round" else
             ", ".join(i_["label"] for i_ in a_["items"] if i_["type"] == t_)) for t_ in a_["types"]])
        a_["inside"] = a_["lo"] * 0.999 <= g_now_b <= a_["hi"] * 1.001 or abs(a_["price"] / g_now_b - 1) <= 0.0015
    here = next((a_ for a_ in areas if a_["inside"]), None)
    above = [a_ for a_ in areas if a_["price"] > g_now_b and a_ is not here][:2]
    below = [a_ for a_ in reversed(areas) if a_["price"] < g_now_b and a_ is not here][:2]

    def area_txt(a_):
        tag = ("a key level where " + and_join([TYPE_NAME[t_] for t_ in a_["types"]]) + " meet"
               if len(a_["types"]) >= 2 else a_["name"])
        return (f"{money(a_['price'])} ({tag}; {(a_['price'] / g_now_b - 1) * 100:+.1f}%, GDXU about ${gx_b(a_['price']):.2f}, "
                f"about {touch(a_['price'])}% chance gold trades there this week)")

    # what happened in the last 24 hours
    t24 = gold_hl.index[-1] - pd.Timedelta(hours=24)
    g24 = float(gold_hl["Close"][gold_hl.index <= t24].iloc[-1]) if (gold_hl.index <= t24).any() else g_now_b
    ch24 = g_now_b / g24 - 1
    last24 = gold_hl[gold_hl.index > t24]
    hi24, lo24 = float(last24["High"].max()), float(last24["Low"].min())
    rng_now = hi24 / lo24 - 1
    rng_norm = float((gdo["h"] / gdo["l"] - 1).iloc[-21:-1].mean()) if gdo is not None else rng_now
    rng_word = "a quiet" if rng_now < 0.7 * rng_norm else "a busy" if rng_now > 1.4 * rng_norm else "a normal"
    happened = (f"Gold is {ch24 * 100:+.1f}% over the last 24 hours, trading between {money(lo24)} and {money(hi24)}, "
                f"{rng_word} range for gold right now (typical day: {rng_norm * 100:.1f}%).")

    # where it stands: levels, six-month swing, moving averages
    stands = f"Gold is at {money(g_now_b)}"
    if here:
        stands += (", right on a key level where " + and_join([TYPE_NAME[t_] for t_ in here["types"]]) + " meet"
                   if len(here["types"]) >= 2 else f", inside a {here['name']}") + f" ({money(here['lo'])} to {money(here['hi'])})"
    stands += "."
    if fib_info:
        H_, L_ = fib_info["high"], fib_info["low"]
        if g_now_b > H_:
            stands += f" Gold is above its six-month high of about {money(H_)}."
        elif g_now_b < L_:
            stands += f" Gold is below its six-month low of about {money(L_)}."
        elif fib_info["down"]:
            rec = (g_now_b - L_) / (H_ - L_)
            stands += (f" Over six months gold fell from about {money(H_)} to {money(L_)} and has since recovered "
                       f"{rec * 100:.0f}% of that fall.")
        else:
            pb = (H_ - g_now_b) / (H_ - L_)
            stands += (f" Over six months gold rose from about {money(L_)} to {money(H_)} and has since given back "
                       f"{pb * 100:.0f}% of that rise.")
    if 50 in ma_info and 200 in ma_info:
        a50, a200 = g_now_b > ma_info[50], g_now_b > ma_info[200]
        stands += (" It is " + ("above" if a50 else "below") + f" its 50-day average ({money(ma_info[50])}), so the short-term trend is "
                   + ("up" if a50 else "down") + ", and " + ("above" if a200 else "below") + f" its 200-day average ({money(ma_info[200])}), "
                   + "so the long-term trend is " + ("still up." if a200 else "down."))

    holds = ("The first test above is " + area_txt(above[0])
             + (f", then {area_txt(above[1])}" if len(above) > 1 else "") + "."
             if above else "No mapped level within 6% above.")
    if brk and brk.get("watchLevel") and not brk.get("open") and above and \
            abs(above[0]["price"] / brk["watchLevel"] - 1) <= 0.003:
        holds += f" An hourly close above {money(brk['watchLevel'])} also triggers a breakout call (experiment)."
    breaks = ("The first floor below is " + area_txt(below[0])
              + (f", then {area_txt(below[1])}" if len(below) > 1 else "") + "."
              if below else "No mapped level within 6% below.")

    watch = []
    if pre_info:
        watch.append(f"GDXU is likely to open around ${pre_info['open']:.2f} ({(pre_info['open'] / price - 1) * 100:+.1f}%) "
                     f"because gold moved {pre_info['goldMove'] * 100:+.1f}% since its close")
    if events:
        watch.append("scheduled releases: " + ", ".join(events) + " (moves are usually bigger around them)")
    if risk_info and risk_info["bucket"] >= 3:
        watch.append(f"gold's last hour was {risk_info['level']}, so GDXU's next hour is likely about {risk_info['mult']:.1f}x normal size")
    if brk and brk.get("open"):
        watch.append(f"an open breakout call: target {money(brk['open']['target'])}, invalidated at {money(brk['open']['stop'])}")
    if big_week:
        watch.append("gold just had one of its strongest weeks; GDXU has tended to give back ground after such weeks (untested)")

    if trend_info:
        backdrop = ("Direction: no tested signal for the next days. The slow backdrop, gold's 12-month trend, is "
                    + ("up, which historically was only a mild tailwind." if trend_info["m12"] > 0
                       else "down, which historically meant weaker months on average."))
    else:
        backdrop = "Direction: no tested signal for the next days."

    if here and here["key"] and here["price"] <= g_now_b * 1.0015:
        head = "Gold is sitting on a well-watched floor."
    elif here and here["key"]:
        head = "Gold is pressing against a well-watched ceiling."
    elif brk and brk.get("open"):
        head = "Gold has broken above yesterday's high."
    elif ch24 > 0.015:
        head = "Gold is rallying."
    elif ch24 < -0.015:
        head = "Gold is under pressure."
    elif above and below and min(abs(above[0]["price"] / g_now_b - 1), abs(below[0]["price"] / g_now_b - 1)) > 0.01:
        head = "Gold is between levels, with no test close by."
    else:
        head = "Gold is trading near its mapped levels."
    brief = {"head": head, "happened": happened, "stands": stands, "holds": holds, "breaks": breaks,
             "watch": watch, "backdrop": backdrop}

    # chart captions
    if live_info:
        pos = "inside" if day_q[0] <= live_info["price"] <= day_q[4] else "outside"
        captions["gdxu"] = (f"GDXU is {live_info['change'] * 100:+.1f}% today at ${live_info['price']:.2f}, {pos} the usual 80% "
                            f"range for today's close.")
    elif pre_info:
        captions["gdxu"] = (f"Before the open: gold has moved {pre_info['goldMove'] * 100:+.1f}% since GDXU's last close, "
                            f"so GDXU is likely to open near ${pre_info['open']:.2f}.")
    else:
        captions["gdxu"] = (f"GDXU closed at ${price:.2f} ({last_move * 100:+.1f}% on the day), "
                            + ("an unusually big day." if bigger_than >= 0.8 else "an ordinary day for GDXU." if bigger_than < 0.5
                               else "a fairly active day."))
    near = here or (below[0] if below and (not above or g_now_b / below[0]["price"] < above[0]["price"] / g_now_b) else
                    (above[0] if above else None))
    if near is not None:
        lvl_ = near["price"]
        tests = int(((last24["Low"] <= lvl_ * 1.0015) & (last24["High"] >= lvl_ * 0.9985)).astype(int).diff().clip(lower=0).sum()
                    + int((last24["Low"].iloc[0] <= lvl_ * 1.0015) and (last24["High"].iloc[0] >= lvl_ * 0.9985)))
        captions["gold"] = (f"Gold {ch24 * 100:+.1f}% in 24 hours. The nearest mapped level is {money(lvl_)} "
                            f"({near['name']}); gold has touched it {tests} separate time{'s' if tests != 1 else ''} in the last day.")
    else:
        captions["gold"] = f"Gold {ch24 * 100:+.1f}% in 24 hours, with no mapped level nearby."
    if fib_info:
        captions["goldDaily"] = (f"Six-month range {money(fib_info['low'])} to {money(fib_info['high'])}; gold is "
                                 f"{(g_now_b / fib_info['high'] - 1) * 100:+.1f}% from the high and "
                                 f"{(g_now_b / fib_info['low'] - 1) * 100:+.1f}% from the low.")
except Exception as err:
    print("Briefing skipped:", err)

# ---------------- package and write ----------------
dh_ = pd.concat([gdxu_open_all, gdxu_hi_all, gdxu_lo_all, close["GDXU"]], axis=1,
                keys=["o", "h", "l", "c"]).loc[:asof].dropna().iloc[-130:]
data = {
    "asof": f"{asof:%A %d %B %Y}",
    "updated": f"{datetime.now(timezone.utc):%d %b %H:%M} UTC",
    "marketOpen": bool(live is not None),
    "weekEnd": f"{next_days[-1]:%A %d %B}",
    "price": price, "vol": float(sig),
    "closeQ": [float(v) for v in close_q], "lowQ": [float(v) for v in low_q], "zq": zq,
    "events": events,
    "daily": [[int(pd.Timestamp(t).tz_localize(NY).timestamp()), round(float(r_.o), 2), round(float(r_.h), 2),
               round(float(r_.l), 2), round(float(r_.c), 2)] for t, r_ in dh_.iterrows()],
    "futureDates": [f"{d:%d %b}" for d in next_days],
    "gold": {"now": float(gold.iloc[-1]), "gvz": float(close["GVZ"].dropna().iloc[-1]) / 100,
             "alpha": float(alpha), "beta": float(beta), "r2": float(r_sq),
             "q": [float(v) for v in np.quantile(resid, [0.10, 0.50, 0.90])],
             "decay": float((LEVERAGE ** 2 - LEVERAGE) / 2 * r.iloc[-20:].var() * SCEN_DAYS),
             "drag": float(daily_drag * SCEN_DAYS), "days": SCEN_DAYS},
    "bigWeek": {"flag": bool(big_week), "ret": this_week, "rank": week_rank},
    "dragYear": float(1 - np.exp(-daily_drag * 252)), "decayDay": float(decay_day),
    "log": scored, "brk": brk,
    "day": {"target": f"{target_day:%A %d %B}", "q": [float(v) for v in day_q],
            "vol": float(sig_d), "events": day_events, "live": live_info, "pre": pre_info,
            "lastDate": f"{asof:%A %d %B}", "lastMove": last_move,
            "biggerThan": bigger_than, "log": dscored},
    "intraday": intraday, "rest": rest_info, "risk": risk_info, "drivers": drivers_info,
    "trend": trend_info, "vrp": vrp_info, "cot": cot_info, "board": board,
    "levels": levels, "lvlStats": lvl_stats, "goldDaily": gold_daily, "brief": brief, "captions": captions,
    "alerts": [{"title": t_, "msg": m_} for _, t_, m_ in alerts_out],
}
page = open("site_template3.html", encoding="utf-8").read()
with open(OUT_FILE, "w", encoding="utf-8") as fh:
    fh.write(page.replace("__DATA__", json.dumps(data)))
print(f"Saved {OUT_FILE}")

# ===== END OF gdxu_pro3.py =====
