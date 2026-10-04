"""
stocks_page2.py
Builds stocks.html (from stocks_template2.html): trend-filtered momentum in large tech stocks,
with Top picks (3-month price ranges, analyst targets, risk checklist) scored live each quarter.
Evidence (stock_research3.py): on the January 2016 Nasdaq-100 list (no hindsight), the top 10% by
12-1 month momentum, held 3 months only while SPY was above its 200-day average, beat the average
member by about 2.8% a quarter after costs (p = 0.012, positive in both halves). Marginal pass.
Runs once a day after the US close (see .github/workflows/stocks.yml).
Paper portfolio: rebalanced at the first run of each calendar quarter; holds QQQ when the filter is off.
"""
import io
import json
import os
import urllib.request
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
UNIV_FILE, PORT_FILE, SENT_FILE = "stocks_universe.csv", "stocks_portfolio.json", "stocks_alerts_sent.json"
OUT, TEMPLATE = "stocks.html", "stocks_template2.html"
PICKS_N, PICKS_LOG = 5, "stocks_picks_log.json"
MKT_Q, EDGE_Q = 0.02, 0.028          # typical market drift per quarter; tested momentum edge per quarter (half is used)
CORE_N, WIDE_N, COST = 10, 25, 0.002
NDX_FALLBACK = """AAPL MSFT NVDA AMZN AVGO META GOOGL GOOG TSLA COST NFLX PLTR ASML TMUS CSCO AMD AZN LIN INTU PEP
ISRG TXN BKNG QCOM AMGN ADBE ARM PDD HON AMAT GILD CMCSA PANW MU ADP LRCX APP MELI KLAC CRWD INTC ADI SBUX
VRTX CEG DASH CTAS MDLZ ORLY FTNT MAR SNPS CDNS PYPL ABNB REGN MRVL ROP MNST ADSK CSX AXON WDAY TTWO AEP
CHTR NXPI PCAR FAST PAYX ROST EA KDP CPRT BKR FANG IDXX VRSK EXC TEAM CCEP XEL ZS CTSH GEHC KHC MCHP DDOG
ODFL CSGP TTD LULU ON DXCM BIIB CDW GFS WBD MDB SHOP MSTR TRI""".split()
NY = "America/New_York"


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (stocks page)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="ignore")


def wiki_table(page, min_rows):
    for tb in pd.read_html(io.StringIO(fetch("https://en.wikipedia.org/wiki/" + page))):
        cols = {str(c).lower().strip(): c for c in tb.columns}
        tc = next((cols[k] for k in ("ticker", "symbol") if k in cols), None)
        nc = next((cols[k] for k in ("company", "security", "name") if k in cols), None)
        if tc is not None and len(tb) >= min_rows:
            return {str(t).replace(".", "-").upper(): (str(tb.loc[i, nc]) if nc is not None else str(t))
                    for i, t in tb[tc].items()}
    raise ValueError("no table")


# ---------------- universe (refreshed at most once a month) ----------------
def build_universe():
    if os.path.exists(UNIV_FILE):
        u = pd.read_csv(UNIV_FILE)
        age = (pd.Timestamp.now() - pd.Timestamp(u["built"].iloc[0])).days
        if age < 31:
            return u
    rows = {}
    try:
        sp = pd.read_csv(io.StringIO(fetch(
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")))
        for t, n in zip(sp[sp.columns[0]], sp[sp.columns[1]]):
            rows.setdefault(str(t).replace(".", "-").upper(), {"name": str(n), "ndx": False})
    except Exception as err:
        print("S&P 500 list unavailable:", err)
    try:
        for t, n in wiki_table("List_of_S%26P_400_companies", 350).items():
            rows.setdefault(t, {"name": n, "ndx": False})
    except Exception as err:
        print("S&P 400 list unavailable:", err)
    try:
        ndx = wiki_table("Nasdaq-100", 90)
    except Exception as err:
        print("Nasdaq-100 list unavailable, using built-in list:", err)
        ndx = {t: t for t in NDX_FALLBACK}
    for t, n in ndx.items():
        rows.setdefault(t, {"name": n, "ndx": True})
        rows[t]["ndx"] = True
    if len(rows) < 100 and os.path.exists(UNIV_FILE):
        return pd.read_csv(UNIV_FILE)
    u = pd.DataFrame([{"ticker": t, **v} for t, v in rows.items()])
    u["built"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    u.to_csv(UNIV_FILE, index=False)
    return u


U = build_universe()
print(f"Universe: {len(U)} stocks, Nasdaq-100: {int(U['ndx'].sum())}")

# ---------------- prices ----------------
start = (pd.Timestamp.now() - pd.Timedelta(days=620)).strftime("%Y-%m-%d")
tick = list(U["ticker"]) + ["SPY", "QQQ"]
parts = []
for i in range(0, len(tick), 100):
    chunk = tick[i:i + 100]
    raw = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=False, group_by="column")
    cl = raw["Close"]
    parts.append(cl.to_frame(chunk[0]) if isinstance(cl, pd.Series) else cl)
px = pd.concat(parts, axis=1).sort_index()
px = px.loc[:, ~px.columns.duplicated()].dropna(how="all")
last_day = px.index[-1]
spy, qqq = px["SPY"].dropna(), px["QQQ"].dropna()
spy200 = float(spy.iloc[-200:].mean())
trend_on = bool(spy.iloc[-1] > spy200)
print(f"Prices to {last_day:%Y-%m-%d}; trend filter {'ON' if trend_on else 'OFF'}")

stocks = px.drop(columns=["SPY", "QQQ"], errors="ignore")
ok = stocks.iloc[-1].notna() & (stocks.count() >= 273)
stocks = stocks.loc[:, ok]
lp = np.log(stocks)
mom = (lp.iloc[-22] - lp.iloc[-253]).dropna()                  # 12-1 month momentum, as tested
r1m = stocks.iloc[-1] / stocks.iloc[-22] - 1
ma200 = stocks.iloc[-200:].mean()
vol = np.log(stocks).diff().iloc[-60:].std() * np.sqrt(252)
names = dict(zip(U["ticker"], U["name"]))
ndx_set = set(U.loc[U["ndx"], "ticker"])


def row(t, rank):
    p = float(stocks[t].iloc[-1])
    v = float(vol[t])
    spark = stocks[t].iloc[-252:].iloc[::5]
    return {"t": t, "name": names.get(t, t), "rank": rank, "price": p, "mom": float(mom[t]), "r1m": float(r1m[t]),
            "vs200": float(p / ma200[t] - 1), "vol": v, "badq": float(np.exp(-1.28 * v * np.sqrt(63 / 252)) - 1),
            "spark": [round(float(x), 2) for x in spark.values]}


ndx_mom = mom[[t for t in mom.index if t in ndx_set]].sort_values(ascending=False)
core = [row(t, i + 1) for i, t in enumerate(ndx_mom.index[:CORE_N])]
wide = [row(t, i + 1) for i, t in enumerate(mom.sort_values(ascending=False).index[:WIDE_N])]
for w in wide:
    w["ndx"] = w["t"] in ndx_set

# earnings dates for the core list (best effort)
for c in core:
    c["earn"] = None
    try:
        cal = yf.Ticker(c["t"]).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            d0 = pd.Timestamp(ed[0] if isinstance(ed, (list, tuple)) else ed)
            if d0 >= last_day - pd.Timedelta(days=1):
                c["earn"] = f"{d0:%a %d %b}"
                c["earnDays"] = int((d0 - last_day).days)
    except Exception:
        pass

# ---------------- paper portfolio ----------------
day = f"{last_day:%Y-%m-%d}"
qkey = f"{last_day.year}Q{(last_day.month - 1) // 3 + 1}"
port = json.load(open(PORT_FILE)) if os.path.exists(PORT_FILE) else None
changes = None


def value_now(pf):
    if pf["mode"] == "momentum":
        rel = [float(px[t].dropna().iloc[-1]) / e for t, e in pf["entry"].items() if t in px and px[t].notna().any()]
        return pf["nav_at_rebal"] * float(np.mean(rel))
    return pf["nav_at_rebal"] * float(qqq.iloc[-1]) / pf["entry"]["QQQ"]


if port is None or port["quarter"] != qkey:
    nav = value_now(port) if port else 100.0
    old = set(port["entry"]) if port else set()
    if trend_on:
        hold = [c["t"] for c in core]
        new_entry = {t: float(stocks[t].iloc[-1]) for t in hold}
        mode = "momentum"
    else:
        new_entry, mode = {"QQQ": float(qqq.iloc[-1])}, "qqq"
    turnover = 1 - len(old & set(new_entry)) / len(new_entry) if old else 1.0
    nav *= (1 - turnover * COST)
    changes = {"in": sorted(set(new_entry) - old), "out": sorted(old - set(new_entry)), "mode": mode, "quarter": qkey}
    hist = port["history"] if port else []
    avg_entry = {t: float(stocks[t].iloc[-1]) for t in ndx_mom.index}
    port = {"quarter": qkey, "mode": mode, "entry": new_entry, "nav_at_rebal": nav, "rebal_date": f"{last_day:%Y-%m-%d}",
            "start": port["start"] if port else f"{last_day:%Y-%m-%d}", "history": hist,
            "qqq_start": port["qqq_start"] if port else float(qqq.iloc[-1]),
            "avg_nav_at_rebal": (port["avg_nav_now"] if port else 100.0), "avg_entry": avg_entry,
            "log": (port["log"] if port else []) + [changes]}
nav_now = value_now(port)
avg_rel = [float(stocks[t].iloc[-1]) / e for t, e in port["avg_entry"].items() if t in stocks]
port["avg_nav_now"] = port["avg_nav_at_rebal"] * float(np.mean(avg_rel)) if avg_rel else port["avg_nav_at_rebal"]
qqq_nav = 100.0 * float(qqq.iloc[-1]) / port["qqq_start"]
day = f"{last_day:%Y-%m-%d}"
port["history"] = [h for h in port["history"] if h[0] != day] + [[day, round(nav_now, 3), round(qqq_nav, 3),
                                                                     round(port["avg_nav_now"], 3)]]
json.dump(port, open(PORT_FILE, "w"))


# ---------------- top picks: 3-month ranges, analyst view, checklist ----------------
from scipy.stats import norm as _n
T_Q = 63 / 252
drift = MKT_Q + (0.5 * EDGE_Q if trend_on else 0.0)
vol_1y = np.log(stocks).diff().iloc[-252:].std() * np.sqrt(252)
picks = []
for c in core[:PICKS_N]:
    t = c["t"]
    sig = float(0.5 * vol[t] + 0.5 * vol_1y[t])            # blend recent and 1-year volatility
    mu = np.log(1 + drift) - 0.5 * sig ** 2 * T_Q
    q = lambda z: float(c["price"] * np.exp(mu + z * sig * np.sqrt(T_Q)))
    pk = {"t": t, "name": c["name"], "rank": c["rank"], "price": c["price"], "mom": c["mom"], "vs200": c["vs200"],
          "r1m": c["r1m"], "earn": c.get("earn"), "earnDays": c.get("earnDays"), "sig": sig,
          "down": q(_n.ppf(0.10)), "base": q(0.0), "up": q(_n.ppf(0.75)), "upBig": q(_n.ppf(0.90))}
    try:
        info = yf.Ticker(t).info
        pk.update({"aTarget": info.get("targetMeanPrice"), "aHigh": info.get("targetHighPrice"),
                   "aLow": info.get("targetLowPrice"), "aN": info.get("numberOfAnalystOpinions"),
                   "aRec": info.get("recommendationKey")})
    except Exception as err:
        print("  analyst data unavailable for", t, type(err).__name__)
    flags = []
    flags.append(["ok" if trend_on else "bad", "Market filter on" if trend_on else "Market filter off: momentum unreliable"])
    if c.get("earnDays") is not None and c["earnDays"] <= 14:
        flags.append(["warn", f"Earnings {c['earn']}: expect a big move either way"])
    else:
        flags.append(["ok", "No earnings in the next 2 weeks" if c.get("earn") else "Earnings date unknown"])
    flags.append(["warn" if c["vs200"] > 0.40 else "ok",
                  (f"{c['vs200'] * 100:.0f}% above its 200-day average" if c["vs200"] >= 0 else
                   f"{-c['vs200'] * 100:.0f}% below its 200-day average") + (": very stretched" if c["vs200"] > 0.40 else "")])
    pk["flags"] = flags
    picks.append(pk)

# log the picks once per quarter and score them three months later
plog = json.load(open(PICKS_LOG)) if os.path.exists(PICKS_LOG) else []
if not any(p["quarter"] == qkey for p in plog):
    for pk in picks:
        plog.append({"quarter": qkey, "date": day, "t": pk["t"], "price": pk["price"], "down": pk["down"], "base": pk["base"],
                     "up": pk["up"], "aTarget": pk.get("aTarget"), "qqq": float(qqq.iloc[-1]), "out": None})
for p in plog:
    if p["out"] is None and pd.Timestamp(p["date"]) + pd.Timedelta(days=91) <= last_day and p["t"] in px:
        when = pd.Timestamp(p["date"]) + pd.Timedelta(days=91)
        pend = float(px[p["t"]].dropna().asof(when)) if px[p["t"]].notna().any() else None
        qend = float(qqq.asof(when))
        seg = px[p["t"]].dropna().loc[p["date"]:when]
        if pend is not None and len(seg):
            p["out"] = {"end": pend, "ret": pend / p["price"] - 1, "qqqRet": qend / p["qqq"] - 1,
                        "aboveUp": bool(pend >= p["up"]), "belowDown": bool(pend <= p["down"]),
                        "hitTarget": bool(p.get("aTarget") and float(seg.max()) >= p["aTarget"])}
json.dump(plog, open(PICKS_LOG, "w"))
done = [p for p in plog if p["out"]]
pick_score = None
if done:
    pick_score = {"n": len(done), "aboveUp": float(np.mean([p["out"]["aboveUp"] for p in done])),
                  "belowDown": float(np.mean([p["out"]["belowDown"] for p in done])),
                  "hitTarget": float(np.mean([p["out"]["hitTarget"] for p in done if p.get("aTarget")]))
                  if any(p.get("aTarget") for p in done) else None,
                  "avgRet": float(np.mean([p["out"]["ret"] for p in done])),
                  "avgQqq": float(np.mean([p["out"]["qqqRet"] for p in done]))}

holdings = []
for t, e in port["entry"].items():
    p = float(px[t].dropna().iloc[-1]) if t in px and px[t].notna().any() else e
    holdings.append({"t": t, "name": names.get(t, "Invesco QQQ" if t == "QQQ" else t), "entry": e, "price": p, "ret": p / e - 1})

# preview: what would change if rebalanced today
preview_in = sorted(set(c["t"] for c in core) - set(port["entry"])) if trend_on else []
preview_out = sorted(set(port["entry"]) - set(c["t"] for c in core)) if trend_on and port["mode"] == "momentum" else []
nq = (pd.Timestamp(last_day.year, ((last_day.month - 1) // 3 + 1) * 3, 1) + pd.offsets.MonthEnd(0))
next_rebal = f"first update after {nq:%d %b %Y}"

# ---------------- alerts (optional) ----------------
topic = os.environ.get("NTFY_TOPIC", "").strip()
sent = json.load(open(SENT_FILE)) if os.path.exists(SENT_FILE) else []
alerts = []
if changes and (changes["in"] or changes["out"]):
    alerts.append((f"rebal-{changes['quarter']}", f"Stocks: {changes['quarter']} rebalance",
                   ("Momentum ON. Buy: " + ", ".join(changes["in"]) + ". Sell: " + (", ".join(changes["out"]) or "none"))
                   if changes["mode"] == "momentum" else "Trend filter OFF: paper portfolio holds QQQ this quarter."))
prev_trend = port.get("trend_last")
if prev_trend is not None and prev_trend != trend_on:
    alerts.append((f"trend-{day}", "Stocks: market filter " + ("ON" if trend_on else "OFF"),
                   f"S&P 500 is {'above' if trend_on else 'below'} its 200-day average. Applies at the next quarterly rebalance."))
port["trend_last"] = trend_on
json.dump(port, open(PORT_FILE, "w"))
for c in core:
    if c["t"] in port["entry"] and c.get("earnDays") is not None and 0 <= c["earnDays"] <= 7:
        alerts.append((f"earn-{c['t']}-{c['earn']}", f"Earnings soon: {c['t']}",
                       f"{c['name']} reports {c['earn']}. Earnings often move a stock 5-10% overnight."))
if topic:
    for key, title, msg in alerts:
        if key in sent:
            continue
        try:
            urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=msg.encode(),
                                                          headers={"Title": title}), timeout=20)
            sent.append(key)
        except Exception as err:
            print("Alert failed:", err)
    json.dump(sent[-300:], open(SENT_FILE, "w"))

data = {
    "updated": f"{datetime.now(timezone.utc):%d %b %H:%M} UTC", "asof": f"{last_day:%A %d %B %Y}",
    "trend": {"on": trend_on, "spy": float(spy.iloc[-1]), "ma200": spy200, "gap": float(spy.iloc[-1] / spy200 - 1)},
    "core": core, "wide": wide, "holdings": holdings, "mode": port["mode"], "quarter": port["quarter"],
    "rebalDate": port["rebal_date"], "start": port["start"], "history": port["history"],
    "previewIn": preview_in, "previewOut": preview_out, "nextRebal": next_rebal,
    "log": port["log"][-8:], "universe": int(len(stocks.columns)), "ndxN": int(len(ndx_mom)),
    "alerts": [{"title": t, "msg": m} for _, t, m in alerts],
    "picks": picks, "pickScore": pick_score, "pickLog": [p for p in plog][-25:], "drift": drift,
}
page = open(TEMPLATE, encoding="utf-8").read()
open(OUT, "w", encoding="utf-8").write(page.replace("__DATA__", json.dumps(data)))
print(f"Saved {OUT}")
# ===== END OF stocks_page2.py =====
