"""Generate the 80 original/edited pairs for the LLM-as-judge check (report Section 6.3).

Run from the repository root (internet needed for the authors' EDA code and, unless --grocery-only, the Emotion dataset):
    python judge/make_judge_sample.py                 # 10 Emotion + 10 grocery originals, 4 operations each = 80 pairs
    python judge/make_judge_sample.py --grocery-only  # grocery originals only (no Hugging Face needed); draws a different
                                                      # grocery sample from the full run, because the random stream differs

Outputs (in the current folder): judge_sheet.csv (pairs; the operation is hidden) and judge_key.csv (pair_id -> operation).
Method: originals of 6-40 words drawn at random (seed 0); grocery reviews flagged by the privacy scan are excluded; each original is
cleaned with the authors' get_only_chars() and edited once per operation (alpha = 0.1) using the authors' functions.
"""
import os, random, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "new_data"))
from eda_new_data import bootstrap, EMOTION_NAMES, load_emotion  # noqa: E402

ALPHA = 0.1
OPS = ["synonym_replacement", "random_insertion", "random_swap", "random_deletion"]
BAD = {477, 514, 632, 715, 778, 802, 900, 925}   # grocery reviews flagged by the privacy scan


def main(grocery_csv="new_data/grocery_reviews_labelled.csv", grocery_only=False):
    bootstrap()
    from eda import get_only_chars, synonym_replacement, random_insertion, random_swap, random_deletion

    def make_edit(clean, op):
        words = clean.split(" ")
        n = max(1, int(ALPHA * len(words)))
        for _ in range(20):                       # regenerate until the edit differs from the original
            if op == "synonym_replacement": out = synonym_replacement(words, n)
            elif op == "random_insertion": out = random_insertion(words, n)
            elif op == "random_swap": out = random_swap(words, n)
            else: out = random_deletion(words, ALPHA)
            edited = " ".join(out)
            if edited != clean:
                break
        return edited

    rng = random.Random(0)
    picks = []
    if not grocery_only:
        (Xtr, ytr), _, _ = load_emotion()
        pool_e = [(i, t, EMOTION_NAMES[l]) for i, (t, l) in enumerate(zip(Xtr, ytr)) if 6 <= len(t.split()) <= 40]
        picks += [("emotion",) + p for p in rng.sample(pool_e, 10)]
    g = pd.read_csv(grocery_csv)
    pool_g = [(int(r.id), r.text, r.label_name) for r in g.itertuples()
              if int(r.id) not in BAD and 6 <= len(r.text.split()) <= 40]
    picks += [("grocery",) + p for p in rng.sample(pool_g, 10)]

    rows = []
    for ds, oid, text, label in picks:
        clean = " ".join(get_only_chars(text).split())
        for op in OPS:
            random.seed(rng.randrange(10 ** 9))
            rows.append(dict(dataset=ds, original_id=oid, label=label, original=clean,
                             edited=make_edit(clean, op), operation=op))
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "pair_id", range(1, len(df) + 1))
    df["rate"] = ""
    for _, grp in df.groupby(["dataset", "operation"]):              # a random half (5 per dataset per operation) for the human rater
        df.loc[rng.sample(list(grp.index), min(5, len(grp))), "rate"] = "yes"
    df["human_label_preserved"] = ""
    df["human_fluency"] = ""
    df[["pair_id", "dataset", "label", "original", "edited", "rate", "human_label_preserved", "human_fluency"]].to_csv("judge_sheet.csv", index=False)
    df[["pair_id", "original_id", "operation"]].to_csv("judge_key.csv", index=False)
    print(len(df), "pairs written;", int((df.rate == "yes").sum()), "marked for the human rater")


if __name__ == "__main__":
    main(grocery_only="--grocery-only" in sys.argv)
