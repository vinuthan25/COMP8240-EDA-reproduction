"""
COMP8240 - Newly constructed dataset: Australian grocery-app reviews (Google Play, AU store).

Pipeline:  collect -> clean -> blind-label sheet (stars hidden) -> human labels -> analysis.
Human label codes:  1 = positive, 0 = negative, 2 = neutral / mixed.
"""
import numpy as np
import pandas as pd

QUERIES = ["Woolworths", "Coles", "Aldi Australia", "Amazon Australia"]   # edit freely
APP_IDS = None   # optional: list of Google Play app ids to use instead of searching


def collect(queries=QUERIES, app_ids=APP_IDS, per_app=400):
    from google_play_scraper import search, reviews, Sort
    targets = []
    if app_ids:
        targets = [(a, a) for a in app_ids]
    else:
        for q in queries:
            hits = search(q, n_hits=3, lang="en", country="au")
            print(f"\nSearch '{q}' (using the first hit):")
            for h in hits:
                print("   ", h["title"], "-", h["appId"])
            if hits:
                targets.append((hits[0]["title"], hits[0]["appId"]))
    rows = []
    for title, app_id in targets:
        res, _ = reviews(app_id, lang="en", country="au", sort=Sort.NEWEST, count=per_app)
        for r in res:
            rows.append({"app": title, "stars": r["score"], "date": r["at"],
                         "text": r["content"]})
        print(f"-> {title}: {len(res)} reviews collected")
    return pd.DataFrame(rows)


def clean(df):
    """Basic preprocessing; prints how many reviews each step removes."""
    log = [("raw", len(df))]
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    nwords = df["text"].str.split().str.len()
    df = df[(nwords >= 4) & (nwords <= 150)]
    log.append(("4-150 words", len(df)))
    df = df.drop_duplicates(subset="text")
    log.append(("duplicates removed", len(df)))
    ascii_ratio = df["text"].apply(lambda s: sum(c.isascii() for c in s) / max(len(s), 1))
    df = df[ascii_ratio > 0.9]
    log.append(("mostly-English / low emoji", len(df)))
    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    print("\nCleaning log:")
    for name, n in log:
        print(f"  {name:<28}{n:>6}")
    return df


def make_label_sheet(df, per_star=40, seed=42):
    """Sample evenly across star ratings; export text WITHOUT stars so labelling is blind."""
    parts = [g.sample(min(len(g), per_star), random_state=seed)
             for _, g in df.groupby("stars")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    to_label = sample[["id", "text"]].copy()
    to_label["label"] = ""          # <- you fill this: 1 = positive, 0 = negative, 2 = neutral/mixed
    key = sample[["id", "app", "stars", "date", "text"]]
    return to_label, key


def analyse(key, labelled):
    """Label distribution, text length stats, and agreement of human labels with star ratings."""
    from sklearn.metrics import cohen_kappa_score
    lab = labelled[["id", "label"]].dropna()
    lab = lab[lab["label"].astype(str).str.strip().isin(["0", "1", "2"])].copy()
    lab["label"] = lab["label"].astype(float).astype(int)
    df = key.merge(lab, on="id")
    df["star_label"] = df["stars"].map(lambda s: 0 if s <= 2 else (1 if s >= 4 else np.nan))
    names = {0: "negative", 1: "positive", 2: "neutral/mixed"}

    print(f"Labelled reviews: {len(df)}")
    print("\nHuman label distribution:")
    print(df["label"].map(names).value_counts().to_string())
    print("\nReviews per app:")
    print(df["app"].value_counts().to_string())
    w = df["text"].str.split().str.len()
    print(f"\nLength (words): mean={w.mean():.1f}, median={w.median():.0f}, "
          f"min={w.min()}, max={w.max()}   (SST-2 average is ~17)")

    m = df[df["star_label"].notna() & df["label"].isin([0, 1])]
    kappa = cohen_kappa_score(m["label"], m["star_label"].astype(int))
    agree = (m["label"] == m["star_label"]).mean()
    print(f"\nHuman label vs star-derived label (1-2 stars=neg, 4-5=pos, 3 excluded; n={len(m)}):")
    print(f"  raw agreement = {agree*100:.1f}%   Cohen's kappa = {kappa:.2f}")

    dis = m[m["label"] != m["star_label"]]
    print(f"\n{len(dis)} reviews where text sentiment disagrees with the star rating, e.g.:")
    for _, r in dis.head(5).iterrows():
        print(f"  [{int(r['stars'])} stars, you said {names[r['label']]}] {r['text'][:150]}")
    return df
