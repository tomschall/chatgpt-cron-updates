#!/usr/bin/env python3
import time
import requests
import pandas as pd
from tabulate import tabulate
from datetime import datetime, timezone

COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# --- Config (minimal & nachvollziehbar) ---
VS_CURRENCY = "usd"
PAGES = 12              # 2*250=500 Coins scannen (für MVP genug)
PER_PAGE = 250
MCAP_MAX = 250_000_000  # < 150 Mio = Moonshot-Range
MCAP_MIN = 2_000_000    # zu kleine Projekte filtern (Illiquidität/Scam-Risiko)
VOL_MIN = 50_000        # min. 24h-Volumen (Liquidität)
TOP_N = 25              # Ausgabegröße

# einfache Gewichte: Summe = 1.0
WEIGHTS = {
    "inv_mcap": 0.35,      # kleinere Market Cap = besser
    "vol_to_mcap": 0.25,   # Liquidität relativ zur Größe
    "mom_7d": 0.20,        # Momentum 7d
    "mom_30d": 0.20,       # Momentum 30d
}


def fetch_markets(page: int) -> list:
    params = {
        "vs_currency": VS_CURRENCY,
        "order": "market_cap_asc",
        "per_page": PER_PAGE,
        "page": page,
        "price_change_percentage": "24h,7d,30d",
        "sparkline": "false",
    }
    HEADERS = {"User-Agent": "MoonshotCopilot/0.1 (+github.com/tomschall)"}
    r = requests.get(f"{COINGECKO_BASE}/coins/markets", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def zscore(series: pd.Series) -> pd.Series:
    # Robuster Z-Score (winsorize light): NaNs->0, clamp extremes
    s = series.fillna(series.median())
    if s.std(ddof=0) == 0:
        return pd.Series([0]*len(s), index=s.index)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-3, 3)


def main():
    all_rows = []
    for p in range(1, PAGES + 1):
        data = fetch_markets(p)
        all_rows.extend(data)
        time.sleep(1.2)  # sanft zur API

    df = pd.DataFrame(all_rows)

    # Grundfilter
    df = df[
        (df["market_cap"].notna()) &
        (df["total_volume"].notna()) &
        (df["market_cap"] >= MCAP_MIN) &
        (df["market_cap"] <= MCAP_MAX) &
        (df["total_volume"] >= VOL_MIN)
    ].copy()

    # Stablecoins raus (heuristisch: name/symbol enthält "usd", Preisbewegung sehr gering)
    maybe_stable = df["name"].str.contains("usd", case=False, na=False) | df["symbol"].str.contains("usd", case=False, na=False)
    low_volatility = df["price_change_percentage_24h_in_currency"].abs().fillna(0) < 0.2
    df = df[~(maybe_stable & low_volatility)].copy()

    if df.empty:
        md = "# Moonshot Report\n\n*(Keine Kandidaten nach aktuellen Filtern gefunden.)*\n"
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(md)
        # leere JSON-Datei anlegen, damit git add nicht scheitert
        with open("report_top.json", "w", encoding="utf-8") as f:
            f.write("[]\n")
        print("No candidates after filter. Wrote empty report and JSON.")
        return

    # Feature Engineering
    df["inv_mcap_feat"] = 1 / df["market_cap"].astype(float)          # kleiner besser
    df["vol_to_mcap_feat"] = (df["total_volume"] / df["market_cap"]).astype(float)
    df["mom_7d_feat"] = df["price_change_percentage_7d_in_currency"].astype(float)
    df["mom_30d_feat"] = df["price_change_percentage_30d_in_currency"].astype(float)

    # Z-Scores
    df["z_inv_mcap"] = zscore(df["inv_mcap_feat"])
    df["z_vol_to_mcap"] = zscore(df["vol_to_mcap_feat"])
    df["z_mom_7d"] = zscore(df["mom_7d_feat"])
    df["z_mom_30d"] = zscore(df["mom_30d_feat"])

    # Weighted Score
    df["moonshot_score"] = (
        WEIGHTS["inv_mcap"] * df["z_inv_mcap"] +
        WEIGHTS["vol_to_mcap"] * df["z_vol_to_mcap"] +
        WEIGHTS["mom_7d"] * df["z_mom_7d"] +
        WEIGHTS["mom_30d"] * df["z_mom_30d"]
    )

    # Sort & select
    cols = [
        "market_cap_rank", "id", "symbol", "name", "current_price", "market_cap",
        "total_volume", "price_change_percentage_7d_in_currency", "price_change_percentage_30d_in_currency",
        "moonshot_score"
    ]
    out = df.sort_values("moonshot_score", ascending=False)[cols].head(TOP_N).copy()

    # Schönere Spalten
    out.rename(columns={
        "market_cap_rank": "rank",
        "current_price": "price_usd",
        "market_cap": "mcap_usd",
        "total_volume": "vol_24h_usd",
        "price_change_percentage_7d_in_currency": "ch_7d_pct",
        "price_change_percentage_30d_in_currency": "ch_30d_pct",
    }, inplace=True)

    # Runden
    for c in ["price_usd", "mcap_usd", "vol_24h_usd"]:
        out[c] = out[c].round(2)
    for c in ["ch_7d_pct", "ch_30d_pct", "moonshot_score"]:
        out[c] = out[c].round(3)

    # Markdown-Report
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"# Moonshot Report\n\nStand: **{ts}**\n\n"
    desc = (
        "Kriterien (MVP): Market Cap 5–150 Mio USD, 24h-Volumen ≥ 100k, Stablecoins ausgeschlossen.\n"
        "Score = gewichtete Z-Scores aus: kleiner MCAP (besser), Volumen/MCAP, Momentum 7d & 30d.\n\n"
        "**Hinweis:** Nur Research/Signale, keine Finanzberatung.\n\n"
    )
    table_md = tabulate(out, headers="keys", tablefmt="github", showindex=False)
    footer = (
        "\n\n*Weights:* "
        f"{WEIGHTS}  \n*Quelle:* CoinGecko `/coins/markets`  \n"
        "*Nächste Ausbaustufe:* Dev-Aktivität (GitHub), Social-Momentum, Narrativ-Erkennung.\n"
    )
    md = header + desc + table_md + footer

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Zusätzlich Rohdaten-Snapshot (optional)
    out.to_json("report_top.json", orient="records", indent=2)

    print(f"Candidates after scoring: {len(out)}")
    print("Wrote report.md and report_top.json")


if __name__ == "__main__":
    main()
