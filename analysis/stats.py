"""Statistics reported in the final report.

Run from the repository root:   python analysis/stats.py
Needs: pandas, numpy, scipy, scikit-learn. No internet required.

1. Replication (report Table 5): mean +/- s.d. over seeds, paired t-test, seeds where EDA won.
2. LLM-as-judge (report Table 9 and Section 6.3): per-operation results and agreement with the human ratings.
3. Grocery labels versus star ratings. NOTE: the public CSV has 194 reviews (five withheld for privacy), so these
   numbers differ slightly from the 91.9% / kappa 0.84 in the report, which were computed on all 199.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

# ---- 1. replication -------------------------------------------------------------------------------
rep = pd.read_csv("replication/results/raw_results.csv")
rep["acc"] = rep.test_accuracy * 100
print("== SST-2 replication (CNN, test accuracy %)")
print(f"{'Train':>8} {'Normal':>14} {'+EDA':>14} {'Diff':>6} {'p':>6} {'Wins':>5}")
for n in ["500", "2000", "5000", "full"]:
    s = rep[rep.N_train.astype(str) == n]
    a = s[s.condition == "normal"].sort_values("seed").acc.values
    b = s[s.condition == "EDA"].sort_values("seed").acc.values
    p = stats.ttest_rel(b, a).pvalue
    print(f"{n:>8} {a.mean():6.1f} +/- {a.std(ddof=1):3.1f} {b.mean():6.1f} +/- {b.std(ddof=1):3.1f} "
          f"{b.mean() - a.mean():+6.2f} {p:6.2f} {int((b > a).sum()):>3}/5")

# ---- 2. LLM-as-judge ------------------------------------------------------------------------------
j = pd.read_csv("judge/judge_results.csv")
print("\n== LLM-as-judge (80 pairs)")
print(j.groupby("operation").agg(n=("pair_id", "size"),
                                 label_kept_pct=("llm_label_preserved", lambda x: 100 * x.mean()),
                                 judge_fluency=("llm_fluency", "mean")).round(2).to_string())
print(f"overall: label kept {100 * j.llm_label_preserved.mean():.0f}%, mean fluency {j.llm_fluency.mean():.2f}")
h = j[j.rated_by_human == "yes"]
hl, jl = h.human_label_preserved.astype(int), h.llm_label_preserved.astype(int)
hf, jf = h.human_fluency.astype(int), h.llm_fluency.astype(int)
print(f"\nHuman vs judge on {len(h)} pairs:")
print(f"  label agreement {100 * (hl == jl).mean():.0f}% ({int((hl == jl).sum())}/{len(h)}), Cohen's kappa {cohen_kappa_score(hl, jl):.2f}")
print(f"  fluency within 1 point {100 * ((hf - jf).abs() <= 1).mean():.0f}%, Spearman {stats.spearmanr(hf, jf).correlation:.2f}, "
      f"mean human {hf.mean():.2f} vs judge {jf.mean():.2f}")
print("  human mean fluency by operation:", h.groupby("operation").human_fluency.mean().round(1).to_dict())

# ---- 3. grocery labels versus star ratings --------------------------------------------------------
d = pd.read_csv("new_data/grocery_reviews_labelled.csv")
star_label = d.stars.map(lambda s: 0 if s <= 2 else (1 if s >= 4 else np.nan))
m = d[star_label.notna() & d.label.isin([0, 1])]
print(f"\n== Grocery labels vs stars on the released {len(d)} reviews (n={len(m)} after excluding 3-star)")
print(f"  agreement {100 * (m.label == star_label[m.index]).mean():.1f}%, "
      f"Cohen's kappa {cohen_kappa_score(m.label, star_label[m.index].astype(int)):.2f}")
print("  label counts:", d.label_name.value_counts().to_dict(),
      "| majority baseline %.1f%%" % (100 * d.label.value_counts(normalize=True).max()))
