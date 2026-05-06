"""
context_adjust.py
-----------------
Corrects shooting quality (SQS) and player value (e_net_rating) for
team context and role dependency.

Inputs (from companion repos):
    data/raw/sqs_rankings.csv      — from nba-sqs
    data/raw/team_context.csv      — from nba-analytics
    data/raw/shot_locations.csv    — from nba-sqs data/raw/

Three correction layers:

    1. Context Premium Adjustment
       Players on good teams have inflated e_net_rating. The context_premium
       from nba-analytics (predicted - actual e_net_rating) isolates how much
       of a player's value is team-manufactured vs individually produced.

    2. Assisted Rate Adjustment
       Players who rarely create their own shots receive PAE credit that
       belongs to their teammates. Assisted rate discounts PAE for players
       whose shot diet is primarily manufactured by others.
       Source: leaguedashptstats (PtMeasureType=Efficiency)

    3. Zone Entropy Penalty
       Players whose shot diet is heavily concentrated in high-PPA zones
       (primarily the rim) receive an entropy penalty on PAE. A diverse
       shot diet indicates genuine shooting range. A concentrated one
       indicates a sheltered role.
       entropy = -Σ (zone_share × log(zone_share))

Final output — Context-Adjusted SQS (SQS_ca):
    SQS_ca = 0.5 × z(PPSA_adj) + 0.5 × z(PAE_adj)

    Where PAE_adj = PAE × assisted_rate_weight × entropy_weight

Outputs:
    outputs/context_adjusted_rankings.csv
    outputs/context_analysis.png

Run: python3 context_adjust.py
"""

import time
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path
from curl_cffi import requests as curl_requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEASON     = "2023-24"
DELAY      = 5.0
RETRY_WAIT = 15.0
MAX_RETRIES = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

# Weight constants
# How much assisted rate discounts PAE
# 0 = no discount, 1 = full discount for 100% assisted players
K_ASSISTED = 0.40

# How much zone entropy discounts PAE
# 0 = no discount, 1 = full discount for zero-entropy players
K_ENTROPY  = 0.30

K_CONTEXT  = 0.25


# ── fetch helper ──────────────────────────────────────────────────────────────

def fetch(url: str, params: dict) -> pd.DataFrame:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = curl_requests.get(
                url, params=params, headers=HEADERS,
                impersonate="chrome110", timeout=90
            )
            r.raise_for_status()
            data = r.json()
            cols = data["resultSets"][0]["headers"]
            rows = data["resultSets"][0]["rowSet"]
            return pd.DataFrame(rows, columns=cols)
        except Exception as e:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)
            else:
                raise


# ── load inputs ───────────────────────────────────────────────────────────────

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for fname in ["sqs_rankings.csv", "team_context.csv", "shot_locations.csv"]:
        if not (RAW / fname).exists():
            raise FileNotFoundError(
                f"data/raw/{fname} not found.\n"
                f"Copy from companion repos first — see README."
            )

    sqs      = pd.read_csv(RAW / "sqs_rankings.csv")
    context  = pd.read_csv(RAW / "team_context.csv")
    shot_loc = pd.read_csv(RAW / "shot_locations.csv")

    # Filter context to 2023-24
    context = context[context["season"] == SEASON].copy()

    print(f"  sqs_rankings.csv     {len(sqs)} players")
    print(f"  team_context.csv     {len(context)} players")
    print(f"  shot_locations.csv   {len(shot_loc)} players")

    return sqs, context, shot_loc


# ── pull assisted rate ────────────────────────────────────────────────────────

def pull_assisted_rate() -> pd.DataFrame:
    """
    Derive assisted rate proxy from catch-and-shoot and pull-up CSVs.
    Catch-and-shoot FGA is a strong proxy for assisted shots.
    Pull-up FGA is a strong proxy for unassisted/self-created shots.

    assisted_rate_proxy = cs_fga / (cs_fga + pu_fga)
    """
    cs_path = Path.home() / "nba-sqs/data/raw/catch_shoot.csv"
    pu_path  = Path.home() / "nba-sqs/data/raw/pull_up.csv"

    cs = pd.read_csv(cs_path)[["PLAYER_NAME", "cs_fga"]]
    pu = pd.read_csv(pu_path)[["PLAYER_NAME", "pu_fga"]]

    df = cs.merge(pu, on="PLAYER_NAME", how="outer").fillna(0)
    total = df["cs_fga"] + df["pu_fga"]
    df["assisted_rate"] = np.where(total > 0, df["cs_fga"] / total, np.nan)
    df = df.rename(columns={"PLAYER_NAME": "player_name"})

    path = RAW / "assisted_rate.csv"
    df.to_csv(path, index=False)
    print(f"  ✓ Derived from catch_shoot + pull_up CSVs ({len(df)} players)")
    return df[["player_name", "assisted_rate"]]


# ── zone entropy ──────────────────────────────────────────────────────────────

def compute_zone_entropy(shot_loc: pd.DataFrame) -> pd.DataFrame:
    """
    For each player compute Shannon entropy of their zone distribution.

    High entropy = diverse shot diet = full PAE credit
    Low entropy  = concentrated in one zone = entropy penalty applied

    Normalize entropy to [0,1] within the sample so the penalty
    is relative to the player pool, not an absolute scale.
    """
    zones = ["ra", "paint", "mid", "lc3", "rc3", "ab3", "corner3"]
    fga_cols = [f"{z}_fga" for z in zones if f"{z}_fga" in shot_loc.columns]

    shot_loc = shot_loc.copy()
    total_fga = shot_loc[fga_cols].sum(axis=1).clip(1)

    entropy_vals = []
    for _, row in shot_loc.iterrows():
        shares = np.array([row[c] for c in fga_cols]) / total_fga[row.name]
        shares = shares[shares > 0]
        e = -np.sum(shares * np.log(shares)) if len(shares) > 0 else 0
        entropy_vals.append(e)

    shot_loc["zone_entropy"] = entropy_vals

    # Normalize to [0,1]
    emin = shot_loc["zone_entropy"].min()
    emax = shot_loc["zone_entropy"].max()
    shot_loc["zone_entropy_norm"] = (
        (shot_loc["zone_entropy"] - emin) / (emax - emin)
    ).fillna(0)

    return shot_loc[["PLAYER_NAME", "zone_entropy", "zone_entropy_norm"]]


# ── PAE adjustment ────────────────────────────────────────────────────────────

def adjust_pae(df: pd.DataFrame) -> pd.DataFrame:
    """
    PAE_adj = PAE × assisted_rate_weight × entropy_weight × context_weight

    assisted_rate_weight = 1 - (assisted_rate × K_ASSISTED)
        Discounts PAE for players whose shots are primarily created by teammates.

    entropy_weight = 1 - ((1 - zone_entropy_norm) × K_ENTROPY)
        Discounts PAE for players concentrated in high-PPA zones.

    context_weight = 1 - (context_premium_norm × K_CONTEXT × zone_entropy_norm)
        Context premium boost is gated by zone entropy.
        High entropy players get full context premium credit.
        Low entropy (rim-runners) get almost none — their role
        dependency is already captured by entropy_weight.
    """
    df = df.copy()

    df["assisted_rate"]      = df["assisted_rate"].fillna(df["assisted_rate"].median())
    df["zone_entropy_norm"]  = df["zone_entropy_norm"].fillna(0.5)
    df["context_premium"]    = df["context_premium"].fillna(0)

    # Normalise context_premium to [-1, 1] range for the weight calculation
    cp_max = df["context_premium"].abs().max()
    df["context_premium_norm"] = df["context_premium"] / cp_max if cp_max > 0 else 0

    df["assisted_rate_weight"] = 1 - (df["assisted_rate"] ** 2 * K_ASSISTED)
    df["entropy_weight"]       = 1 - ((1 - df["zone_entropy_norm"]) ** 2 * K_ENTROPY)

    # Context weight gated by zone entropy
    # Players with high entropy get full context premium credit
    # Players with low entropy (rim-runners) get almost none
    # Only penalise players on good teams — don't boost players on bad teams
    # Clip context_premium_norm at 0 so negative values have no effect on PAE
    df["context_weight"] = (1 - (df["context_premium_norm"].clip(lower=0) * K_CONTEXT * df["zone_entropy_norm"])
    )

    df["pae_adj"] = (
        df["pae"]
        * df["assisted_rate_weight"]
        * df["entropy_weight"]
        * df["context_weight"]
    )
    # Role score flag — sheltered finishers
    df["role_score"] = 0.5 * df["assisted_rate"] + 0.5 * (1 - df["zone_entropy_norm"])
    df["sheltered_finisher"] = df["role_score"] > 0.75   

    return df


# ── context premium adjustment ────────────────────────────────────────────────

def apply_context_premium(df: pd.DataFrame) -> pd.DataFrame:
    """
    context_premium = actual e_net_rating - predicted e_net_rating
    (from nba-analytics Ridge model)

    Positive context_premium = player is on a good team,
    their e_net_rating is inflated by team quality.

    context_adjusted_rating = actual - context_premium
    This isolates the individual contribution from team effects.
    """
    df = df.copy()
    df["context_premium"]        = df["context_premium"].fillna(0)
    df["context_adjusted_rating"] = df["actual"] - df["context_premium"]
    return df


# ── SQS_ca ────────────────────────────────────────────────────────────────────

def compute_sqs_ca(df: pd.DataFrame) -> pd.DataFrame:
    """
    SQS_ca = 0.5 × z(PPSA_adj) + 0.5 × z(PAE_adj)

    Uses the same structure as SQS but with PAE replaced by PAE_adj,
    which accounts for assisted rate and zone entropy.
    """
    valid = df["ppsa_adj"].notna() & df["pae_adj"].notna()
    df.loc[valid, "z_ppsa_adj_ca"] = stats.zscore(df.loc[valid, "ppsa_adj"])
    df.loc[valid, "z_pae_adj"]     = stats.zscore(df.loc[valid, "pae_adj"])
    df.loc[valid, "sqs_ca"]        = (
        0.5 * df.loc[valid, "z_ppsa_adj_ca"] +
        0.5 * df.loc[valid, "z_pae_adj"]
    )
    return df


# ── validate ──────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame):
    sub = df.dropna(subset=["sqs", "sqs_ca", "context_adjusted_rating"])

    print(f"\n{'─'*58}")
    print("  VALIDATION vs context_adjusted_rating")
    print(f"{'─'*58}")
    print(f"  Sample: {len(sub)} players\n")

    for col, label in [
        ("sqs",    "SQS     (original)  "),
        ("sqs_ca", "SQS_ca  (adjusted)  "),
    ]:
        r, p = stats.pearsonr(sub[col], sub["context_adjusted_rating"])
        print(f"  {label}  r = {r:.3f}   p = {p:.4f}")


# ── surface findings ──────────────────────────────────────────────────────────

def surface_findings(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["sqs", "sqs_ca"]).copy()

    sub["rank_sqs"]    = sub["sqs"].rank(ascending=False).astype(int)
    sub["rank_sqs_ca"] = sub["sqs_ca"].rank(ascending=False).astype(int)
    sub["rank_delta"]  = sub["rank_sqs"] - sub["rank_sqs_ca"]
    # Positive = player ranked higher under original SQS
    # Negative = player ranked higher under context-adjusted SQS

    print(f"\n{'─'*58}")
    print("  TOP 20 by SQS_ca  (context adjusted)")
    print(f"{'─'*58}")
    cols = ["player_name", "sqs_ca", "sqs", "pae", "pae_adj",
            "assisted_rate", "zone_entropy_norm", "context_premium",
            "rank_sqs_ca", "rank_sqs", "rank_delta"]
    print(sub.sort_values("rank_sqs_ca")[cols].head(20).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'─'*58}")
    print("  BIGGEST DOWNWARD ADJUSTMENTS")
    print("  High SQS → lower SQS_ca  (context/role inflated)")
    print(f"{'─'*58}")
    down = sub[sub["rank_sqs_ca"] > sub["rank_sqs"]].copy()
    down["rank_delta"] = down["rank_sqs_ca"] - down["rank_sqs"]
    down = down.nlargest(15, "rank_delta")
    print(down[["player_name", "sqs", "sqs_ca", "assisted_rate",
                "zone_entropy_norm", "context_premium",
                "rank_sqs", "rank_sqs_ca", "rank_delta"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'─'*58}")
    print("  BIGGEST UPWARD ADJUSTMENTS")
    print("  Low SQS → higher SQS_ca  (undervalued independent shooters)")
    print(f"{'─'*58}")
    up = sub[sub["rank_sqs_ca"] < sub["rank_sqs"]].copy()
    up["rank_delta"] = up["rank_sqs"] - up["rank_sqs_ca"]
    up = up.nlargest(15, "rank_delta")
    print(up[["player_name", "sqs", "sqs_ca", "assisted_rate",
              "zone_entropy_norm", "context_premium",
              "rank_sqs", "rank_sqs_ca", "rank_delta"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    return sub


# ── plots ─────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame):
    sub = df.dropna(subset=["sqs", "sqs_ca", "context_adjusted_rating",
                             "assisted_rate", "zone_entropy_norm",
                             "context_premium"])

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    AXIS_BG = "#1a1d27"
    C1, C2, C3, C4 = "#4fc3f7", "#f06292", "#81c784", "#ffb74d"
    TEXT, GRID = "#e0e0e0", "#2e3244"

    def style(ax, title):
        ax.set_facecolor(AXIS_BG)
        ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=9)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    def annotate_top(ax, xcol, ycol, rank_col, n=6):
        for _, row in sub.nsmallest(n, rank_col).iterrows():
            ax.annotate(
                row["player_name"].split()[-1],
                (row[xcol], row[ycol]),
                fontsize=6.5, color=TEXT, alpha=0.85,
                xytext=(4, 2), textcoords="offset points"
            )

    # P1: SQS vs context_adjusted_rating
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(sub["sqs"], sub["context_adjusted_rating"],
               alpha=0.5, s=18, color=C1, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["sqs"], sub["context_adjusted_rating"])
    xs = np.linspace(sub["sqs"].min(), sub["sqs"].max(), 100)
    ax.plot(xs, m * xs + b, color=C1, lw=1.5)
    ax.set_xlabel("SQS  (original)")
    ax.set_ylabel("context_adjusted_rating")
    style(ax, f"SQS vs context-adjusted rating  (r={r:.3f})")

    # P2: SQS_ca vs context_adjusted_rating
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(sub["sqs_ca"], sub["context_adjusted_rating"],
               alpha=0.5, s=18, color=C2, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["sqs_ca"], sub["context_adjusted_rating"])
    xs = np.linspace(sub["sqs_ca"].min(), sub["sqs_ca"].max(), 100)
    ax.plot(xs, m * xs + b, color=C2, lw=1.5)
    ax.set_xlabel("SQS_ca  (context adjusted)")
    ax.set_ylabel("context_adjusted_rating")
    style(ax, f"SQS_ca vs context-adjusted rating  (r={r:.3f})")

    # P3: Rank delta — who moves and why
    ax = fig.add_subplot(gs[0, 2])
    rank_delta = sub["rank_sqs"] - sub["rank_sqs_ca"]
    colors3 = [C3 if v > 0 else C1 for v in rank_delta]
    ax.scatter(sub["context_premium"], rank_delta,
               c=colors3, alpha=0.5, s=18, linewidths=0)
    ax.axhline(0, color=TEXT, lw=0.8, alpha=0.4)
    ax.set_xlabel("context_premium  (team inflation)")
    ax.set_ylabel("rank delta  (SQS rank − SQS_ca rank)")
    # Annotate biggest movers
    top_down = sub.nlargest(4, "rank_delta")
    top_up   = sub.nsmallest(4, "rank_delta")
    for _, row in pd.concat([top_down, top_up]).iterrows():
        delta = row["rank_sqs"] - row["rank_sqs_ca"]
        ax.annotate(
            row["player_name"].split()[-1],
            (row["context_premium"], delta),
            fontsize=6.5, color=TEXT, alpha=0.85,
            xytext=(4, 2), textcoords="offset points"
        )
    style(ax, "Who moves — driven by context premium")

    # P4: Assisted rate vs PAE adjustment
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(sub["assisted_rate"], sub["pae"] - sub["pae_adj"],
               alpha=0.5, s=18, color=C4, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["assisted_rate"],
                                    sub["pae"] - sub["pae_adj"])
    xs = np.linspace(sub["assisted_rate"].min(), sub["assisted_rate"].max(), 100)
    ax.plot(xs, m * xs + b, color=C4, lw=1.5)
    ax.set_xlabel("Assisted rate")
    ax.set_ylabel("PAE discount  (PAE − PAE_adj)")
    style(ax, f"Assisted rate drives PAE discount  (r={r:.3f})")

    # P5: Zone entropy vs PAE adjustment
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(sub["zone_entropy_norm"], sub["pae"] - sub["pae_adj"],
               alpha=0.5, s=18, color=C3, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["zone_entropy_norm"],
                                    sub["pae"] - sub["pae_adj"])
    xs = np.linspace(sub["zone_entropy_norm"].min(),
                     sub["zone_entropy_norm"].max(), 100)
    ax.plot(xs, m * xs + b, color=C3, lw=1.5)
    ax.set_xlabel("Zone entropy  (shot diet diversity)")
    ax.set_ylabel("PAE discount  (PAE − PAE_adj)")
    style(ax, f"Zone entropy vs PAE discount  (r={r:.3f})")

    # P6: SQS vs SQS_ca — how much did the adjustment move things?
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(sub["sqs"], sub["sqs_ca"],
               alpha=0.5, s=18, color=C1, linewidths=0)
    m, b, r, *_ = stats.linregress(sub["sqs"], sub["sqs_ca"])
    xs = np.linspace(sub["sqs"].min(), sub["sqs"].max(), 100)
    ax.plot(xs, m * xs + b, color=C1, lw=1.5)
    # Identity line
    lims = [min(sub["sqs"].min(), sub["sqs_ca"].min()),
            max(sub["sqs"].max(), sub["sqs_ca"].max())]
    ax.plot(lims, lims, color=TEXT, lw=0.8, alpha=0.4, ls="--")
    ax.set_xlabel("SQS  (original)")
    ax.set_ylabel("SQS_ca  (adjusted)")
    style(ax, f"SQS vs SQS_ca  (r={r:.3f})  — adjustment magnitude")

    fig.suptitle(
        "Context-Adjusted Shooting Quality  ·  2023-24 NBA",
        color=TEXT, fontsize=14, fontweight="bold", y=0.98
    )

    path = OUT / "context_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  ✓ Chart saved: {path}")


# ── save ──────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame):
    cols = [
        "player_name", "team_abbreviation", "sqs_ca", "sqs",
        "ppsa_adj", "pae", "pae_adj", "assisted_rate",
        "zone_entropy_norm", "role_score", "sheltered_finisher",
        "context_premium", "context_adjusted_rating", "actual",
        "rank_sqs_ca", "rank_sqs", "rank_delta",
    ]
    cols = [c for c in cols if c in df.columns]
    out  = df[cols].dropna(subset=["sqs_ca"]).sort_values(
        "sqs_ca", ascending=False).copy()
    out["rank_sqs_ca"] = range(1, len(out) + 1)

    path = OUT / "context_adjusted_rankings.csv"
    out.to_csv(path, index=False)
    print(f"  ✓ Rankings saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"NBA Context-Adjusted Shooting Quality  ·  {SEASON}\n")

    print("── Loading inputs...")
    sqs, context, shot_loc = load_inputs()

    print("\n── Pulling assisted rate...")
    assisted = pull_assisted_rate()

    print("\n── Computing zone entropy...")
    entropy_df = compute_zone_entropy(shot_loc)

    print("\n── Merging datasets...")
    # Normalise player names for merging
    for d in [sqs, context, assisted, entropy_df]:
        key = "player_name" if "player_name" in d.columns else "PLAYER_NAME"
        d[key] = d[key].str.strip()

    df = sqs.merge(
        context[["player_name", "context_premium", "actual",
                 "predicted", "team_abbreviation"]],
        on="player_name", how="left"
    )
    df = df.merge(
        assisted[["player_name", "assisted_rate"]],
        on="player_name", how="left"
    )
    df = df.merge(
        entropy_df.rename(columns={"PLAYER_NAME": "player_name"}),
        on="player_name", how="left"
    )

    print(f"  Merged sample: {df['sqs'].notna().sum()} players with SQS")
    print(f"  Context premium available: {df['context_premium'].notna().sum()}")
    print(f"  Assisted rate available:   {df['assisted_rate'].notna().sum()}")

    print("\n── Adjusting PAE for assisted rate and zone entropy...")
    df = adjust_pae(df)

    print("\n── Applying context premium...")
    df = apply_context_premium(df)

    print("\n── Computing SQS_ca...")
    df = compute_sqs_ca(df)

    validate(df)
    ranked = surface_findings(df)
    plot(ranked)
    save(ranked)

    print("\n✓ Done.")
    print("  outputs/context_analysis.png")
    print("  outputs/context_adjusted_rankings.csv")
