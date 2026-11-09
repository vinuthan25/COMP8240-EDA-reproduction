"""
COMP8240 final report - EDA (Wei & Zou, EMNLP-IJCNLP 2019) applied to NEW data.

Datasets
  1. Emotion (Saravia et al., 2018): 20,000 tweets, 6 emotions, split 16,000 / 2,000 / 2,000.
  2. Grocery-app reviews (built for this project): 199 labelled Google Play AU reviews
     (Woolworths, Coles); labels negative / positive / mixed.

Method: identical to the SST-2 replication (eda_replication.py): the authors' eda() code
for augmentation, and the paper's CNN (static GloVe 300d -> 128 filters of width 5 ->
global max-pool -> dense 20 -> softmax), trained with and without EDA.
EDA settings follow the paper's Table 3 (alpha, n_aug per training size).

Metrics: accuracy (as in the paper) plus macro-F1 and F1 on the rarest class, because both
datasets have uneven classes.
"""
import os, sys, random, subprocess, time
import numpy as np

# Paper Table 3: N_train -> (alpha, n_aug). Small datasets use the N = 500 setting.
SETTINGS = {500: (0.05, 16), 2000: (0.05, 8), 5000: (0.1, 4), "full": (0.1, 4)}
EMOTION_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
GROCERY_NAMES = ["negative", "positive", "mixed"]


# ------------------------------------------------------------------ setup / EDA
def bootstrap():
    """Fetch the authors' EDA code and the WordNet data it needs."""
    if not os.path.exists("eda_nlp"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/jasonwei20/eda_nlp.git"], check=True)
    code_dir = os.path.join("eda_nlp", "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def augment(X, y, alpha, naug, seed):
    """Apply the authors' eda() to every sentence; output = augmented + original."""
    from eda import eda
    random.seed(seed)
    Xa, ya = [], []
    for s, label in zip(X, y):
        for a in eda(s, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=naug):
            Xa.append(a)
            ya.append(label)
    return Xa, ya


def show_examples(X, k=3, seed=0):
    """Print a few real augmentations (for the report and the demo)."""
    from eda import eda
    random.seed(seed)
    for s in X[:k]:
        print("ORIGINAL :", s)
        for a in eda(s, 0.1, 0.1, 0.1, 0.1, num_aug=4)[:4]:
            print("  AUG    :", a)
        print()


# ------------------------------------------------------------------ data loading
def load_emotion():
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion")
    get = lambda s: (list(ds[s]["text"]), [int(v) for v in ds[s]["label"]])
    return get("train"), get("validation"), get("test")


def load_grocery(path="grocery_reviews_labelled.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    return df["text"].astype(str).tolist(), [int(v) for v in df["label"]]


def data_snippets(grocery_path="grocery_reviews_labelled.csv", k=5, outdir="results_new"):
    """Print (and save) a few raw rows of each dataset: the 'snippet' figures for the report."""
    os.makedirs(outdir, exist_ok=True)
    lines = []
    (Xtr, ytr), _, _ = load_emotion()
    lines.append("EMOTION (training set)")
    for i in range(k):
        lines.append(f"  [{EMOTION_NAMES[ytr[i]]}] {Xtr[i]}")
    Xg, yg = load_grocery(grocery_path)
    lines.append("\nGROCERY-APP REVIEWS")
    for i in range(k):
        lines.append(f"  [{GROCERY_NAMES[yg[i]]}] {Xg[i][:160]}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(outdir, "snippets.txt"), "w").write(txt)


# ------------------------------------------------------------------ models
def train_eval_lr(Xtr, ytr, Xdv, ydv, Xte, yte, seed, kv, n_classes, maxlen):
    """Bag-of-words baseline. Only used as a fast smoke test of the pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)
    return clf.predict(vec.transform(Xte))


def train_eval_cnn(Xtr, ytr, Xdv, ydv, Xte, yte, seed, kv, n_classes, maxlen):
    """The paper's CNN. Returns the predicted labels for the test set."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = {"<pad>": 0}
    for s in Xtr + Xdv + Xte:
        for w in s.lower().split():
            if w not in vocab:
                vocab[w] = len(vocab)
    E = np.zeros((len(vocab), 300), dtype=np.float32)
    for w, i in vocab.items():
        if w in kv:
            E[i] = kv[w]

    def enc(texts):
        arr = np.zeros((len(texts), maxlen), dtype=np.int64)
        for r, s in enumerate(texts):
            ids = [vocab[w] for w in s.lower().split()][:maxlen]
            arr[r, :len(ids)] = ids
        return torch.tensor(arr)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(E), freeze=True, padding_idx=0)
            self.conv = nn.Conv1d(300, 128, 5)
            self.fc1 = nn.Linear(128, 20)
            self.out = nn.Linear(20, n_classes)

        def forward(self, x):
            h = self.emb(x).transpose(1, 2)
            h = torch.relu(self.conv(h)).max(dim=2).values
            return self.out(torch.relu(self.fc1(h)))

    model = CNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    xtr, ytr_t = enc(Xtr).to(dev), torch.tensor(ytr).to(dev)
    xdv, ydv_t = enc(Xdv).to(dev), torch.tensor(ydv).to(dev)
    xte = enc(Xte).to(dev)

    def dev_loss():
        model.eval()
        with torch.no_grad():
            return lossf(model(xdv), ydv_t).item()

    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(40):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(xtr[b]), ytr_t[b]).backward()
            opt.step()
        vl = dev_loss()
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:          # early stopping, patience 3 (as in the paper)
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(xte).argmax(1).cpu().numpy()


def setup_model(model):
    if model == "cnn":
        import gensim.downloader as api
        print("Loading GloVe 300d (first run downloads ~380 MB)...")
        return api.load("glove-wiki-gigaword-300"), train_eval_cnn
    return None, train_eval_lr


# ------------------------------------------------------------------ metrics / summary
def metrics(y_true, y_pred, rare_class):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "acc": accuracy_score(y_true, y_pred) * 100,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "rare_f1": f1_score(y_true, y_pred, labels=[rare_class], average=None, zero_division=0)[0] * 100,
    }


COLS = ["dataset", "size", "seed", "condition", "train_sents", "acc", "macro_f1", "rare_f1"]


def summarise(rows, outdir, tag):
    import pandas as pd
    from scipy import stats
    df = pd.DataFrame(rows, columns=COLS)
    df["size"] = df["size"].astype(str)
    df.to_csv(os.path.join(outdir, f"raw_{tag}.csv"), index=False)
    out = []
    for size in dict.fromkeys(df["size"]):
        g = df[df["size"] == size]
        for metric in ("acc", "macro_f1", "rare_f1"):
            a = g[g.condition == "normal"].sort_values("seed")[metric].values
            b = g[g.condition == "EDA"].sort_values("seed")[metric].values
            p = stats.ttest_rel(b, a).pvalue if len(a) > 1 else float("nan")
            out.append((size, metric, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1), b.mean() - a.mean(), p))
    s = pd.DataFrame(out, columns=["size", "metric", "normal_mean", "normal_sd", "eda_mean", "eda_sd", "diff", "paired_t_p"])
    s.to_csv(os.path.join(outdir, f"summary_{tag}.csv"), index=False)
    print(f"\n=== SUMMARY: {tag} (mean over seeds/splits, %; diff = EDA - normal) ===")
    print(s.round(2).to_string(index=False))
    return df, s


def plot_summary(df, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = list(dict.fromkeys(df["size"]))
    x = np.arange(len(sizes))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in zip(axes, ("acc", "macro_f1"), ("Accuracy (%)", "Macro-F1 (%)")):
        for cond, mk in (("normal", "o"), ("EDA", "s")):
            m = [df[(df["size"] == s) & (df.condition == cond)][metric].mean() for s in sizes]
            sd = [df[(df["size"] == s) & (df.condition == cond)][metric].std(ddof=1) for s in sizes]
            ax.errorbar(x, m, sd, marker=mk, capsize=3, label="With EDA" if cond == "EDA" else "Normal")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes)
        ax.set_xlabel("Training sentences")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


# ------------------------------------------------------------------ experiments
def run_grocery(csv_path="grocery_reviews_labelled.csv", splits=10, model="cnn", outdir="results_new"):
    """EDA on the grocery-app reviews: repeated stratified 80/20 splits."""
    from sklearn.model_selection import train_test_split
    bootstrap()
    os.makedirs(outdir, exist_ok=True)
    X, y = load_grocery(csv_path)
    print(f"Grocery reviews: {len(X)} | label counts: {np.bincount(y).tolist()} (negative, positive, mixed)")
    kv, train_fn = setup_model(model)
    alpha, naug = SETTINGS[500]       # small data: use the paper's recommendation for N = 500
    rows = []
    for seed in range(splits):
        Xa_, Xte, ya_, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        Xtr, Xdv, ytr, ydv = train_test_split(Xa_, ya_, test_size=0.15, stratify=ya_, random_state=seed)
        Xaug, yaug = augment(Xtr, ytr, alpha, naug, seed)
        for cond, (Xc, yc) in (("normal", (Xtr, ytr)), ("EDA", (Xaug, yaug))):
            t0 = time.time()
            preds = train_fn(Xc, yc, Xdv, ydv, Xte, yte, seed, kv, 3, 100)
            m = metrics(yte, preds, 2)            # rarest class = mixed
            rows.append(("grocery", "all", seed, cond, len(Xc), m["acc"], m["macro_f1"], m["rare_f1"]))
            print(f"split={seed} {cond:>6} (train={len(Xc):>5}, test={len(Xte)}) acc={m['acc']:.1f} "
                  f"macroF1={m['macro_f1']:.1f} mixedF1={m['rare_f1']:.1f} [{time.time()-t0:.0f}s]")
    return summarise(rows, outdir, "grocery")


def run_emotion(sizes=(500, 2000, 5000), seeds=5, model="cnn", outdir="results_new"):
    """EDA on Emotion: random training subsets, official validation/test splits."""
    bootstrap()
    os.makedirs(outdir, exist_ok=True)
    (Xtr, ytr), (Xdv, ydv), (Xte, yte) = load_emotion()
    print(f"Emotion loaded: train={len(Xtr)} validation={len(Xdv)} test={len(Xte)}")
    kv, train_fn = setup_model(model)
    rows = []
    for size in sizes:
        alpha, naug = SETTINGS[size]
        n = len(Xtr) if size == "full" else size
        for seed in range(seeds):
            idx = list(range(len(Xtr)))
            random.Random(seed).shuffle(idx)
            sub = idx[:n]
            Xs, ys = [Xtr[i] for i in sub], [ytr[i] for i in sub]
            Xaug, yaug = augment(Xs, ys, alpha, naug, seed)
            for cond, (Xc, yc) in (("normal", (Xs, ys)), ("EDA", (Xaug, yaug))):
                t0 = time.time()
                preds = train_fn(Xc, yc, Xdv, ydv, Xte, yte, seed, kv, 6, 70)
                m = metrics(yte, preds, 5)        # rarest class = surprise
                rows.append(("emotion", size, seed, cond, len(Xc), m["acc"], m["macro_f1"], m["rare_f1"]))
                print(f"N={size!s:>5} seed={seed} {cond:>6} (train={len(Xc):>6}) acc={m['acc']:.1f} "
                      f"macroF1={m['macro_f1']:.1f} surpriseF1={m['rare_f1']:.1f} [{time.time()-t0:.0f}s]")
    df, s = summarise(rows, outdir, "emotion")
    plot_summary(df, f"EDA on Emotion (CNN, {seeds} seeds)", os.path.join(outdir, "emotion_results.png"))
    return df, s
