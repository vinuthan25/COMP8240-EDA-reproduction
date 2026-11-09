"""
COMP8240 - Reproducing EDA (Wei & Zou, EMNLP-IJCNLP 2019) on SST-2.

Replicates the paper's core experiment: train a text CNN on random subsets of the
SST-2 training set (N = 500, 2000, 5000, full), with and without EDA augmentation,
using the paper's recommended EDA settings (Table 3), and compare test accuracy.

Setup follows the paper's Section 9.1 CNN: 128 filters of width 5 -> global max pool
-> dense(20, ReLU) -> softmax, Adam, early stopping (patience 3), static GloVe 300d.

Documented differences from the original: PyTorch instead of Keras; GloVe 6B (wiki+
gigaword) instead of GloVe 840B; the standard sentence-level SST-2 split
(6,920 / 872 / 1,821) instead of the paper's own split (7,447 / - / 1,752); a dev set
is used for early stopping.
"""
import os, sys, random, subprocess, urllib.request, time, csv
import numpy as np

BASE = "https://raw.githubusercontent.com/AcademiaSinicaNLPLab/sentiment_dataset/master/data/"
FILES = {"train": "stsa.binary.train", "dev": "stsa.binary.dev", "test": "stsa.binary.test"}
# Paper Table 3 (recommended usage parameters): N_train -> (alpha, n_aug)
SETTINGS = {500: (0.05, 16), 2000: (0.05, 8), 5000: (0.1, 4), "full": (0.1, 4)}
# Paper Table 2, CNN row, averaged over the paper's FIVE datasets (not SST-2 alone)
PAPER_CNN_AVG = {500: (78.6, 80.7), 2000: (85.6, 86.4), 5000: (87.7, 88.3), "full": (88.3, 88.8)}
MAXLEN = 60


def bootstrap():
    """Fetch the authors' EDA code and the WordNet data it needs."""
    if not os.path.exists("eda_nlp"):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/jasonwei20/eda_nlp.git"], check=True)
    sys.path.insert(0, os.path.join("eda_nlp", "code"))
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)


def load_split(name):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", FILES[name])
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + FILES[name], path)
    X, y = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                y.append(int(line[0]))
                X.append(line[2:].strip())
    return X, y


def augment(X, y, alpha, naug, seed):
    """Apply the authors' eda() to every sentence; output = augmented + original."""
    from eda import eda
    random.seed(seed)
    Xa, ya = [], []
    for s, label in zip(X, y):
        for a in eda(s, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                     p_rd=alpha, num_aug=naug):
            Xa.append(a)
            ya.append(label)
    return Xa, ya


def show_examples(X, k=3):
    """Print a few real augmentations (useful to show on screen in the video)."""
    from eda import eda
    random.seed(0)
    for s in X[:k]:
        print("ORIGINAL :", s)
        for a in eda(s, 0.1, 0.1, 0.1, 0.1, num_aug=4)[:4]:
            print("  AUG    :", a)
        print()


# ---------------------------------------------------------------- models
def train_eval_lr(Xtr, ytr, Xdv, ydv, Xte, yte, seed, kv):
    """Quick bag-of-words baseline. Only used as a fast smoke test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(Xtr), ytr)
    return float((clf.predict(vec.transform(Xte)) == np.array(yte)).mean())


def train_eval_cnn(Xtr, ytr, Xdv, ydv, Xte, yte, seed, kv):
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
        arr = np.zeros((len(texts), MAXLEN), dtype=np.int64)
        for r, s in enumerate(texts):
            ids = [vocab[w] for w in s.lower().split()][:MAXLEN]
            arr[r, :len(ids)] = ids
        return torch.tensor(arr)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding.from_pretrained(torch.tensor(E), freeze=True, padding_idx=0)
            self.conv = nn.Conv1d(300, 128, 5)
            self.fc1 = nn.Linear(128, 20)
            self.out = nn.Linear(20, 2)

        def forward(self, x):
            h = self.emb(x).transpose(1, 2)
            h = torch.relu(self.conv(h)).max(dim=2).values
            return self.out(torch.relu(self.fc1(h)))

    model = CNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    xtr, ytr_t = enc(Xtr).to(dev), torch.tensor(ytr).to(dev)
    xdv, ydv_t = enc(Xdv).to(dev), torch.tensor(ydv).to(dev)
    xte, yte_t = enc(Xte).to(dev), torch.tensor(yte).to(dev)

    def evaluate(x, y):
        model.eval()
        with torch.no_grad():
            logits = model(x)
            return lossf(logits, y).item(), (logits.argmax(1) == y).float().mean().item()

    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(40):
        model.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            lossf(model(xtr[b]), ytr_t[b]).backward()
            opt.step()
        vl, _ = evaluate(xdv, ydv_t)
        if vl < best_loss - 1e-4:
            best_loss, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:          # early stopping, patience 3 (as in the paper)
                break
    model.load_state_dict(best_state)
    return evaluate(xte, yte_t)[1]


# ---------------------------------------------------------------- experiment
def run(sizes=(500, 2000, 5000, "full"), seeds=5, model="cnn", outdir="results"):
    bootstrap()
    os.makedirs(outdir, exist_ok=True)
    Xtr, ytr = load_split("train")
    Xdv, ydv = load_split("dev")
    Xte, yte = load_split("test")
    print(f"SST-2 loaded: train={len(Xtr)} dev={len(Xdv)} test={len(Xte)}\n")
    print("=== Example EDA output ===")
    show_examples(Xtr)

    kv, train_fn = None, train_eval_lr
    if model == "cnn":
        import gensim.downloader as api
        print("Loading GloVe 300d (first run downloads ~380 MB)...")
        kv = api.load("glove-wiki-gigaword-300")
        train_fn = train_eval_cnn

    rows = []
    for size in sizes:
        alpha, naug = SETTINGS[size]
        n = len(Xtr) if size == "full" else size
        for seed in range(seeds):
            idx = list(range(len(Xtr)))
            random.Random(seed).shuffle(idx)
            sub = idx[:n]
            Xs, ys = [Xtr[i] for i in sub], [ytr[i] for i in sub]
            Xa, ya = augment(Xs, ys, alpha, naug, seed)
            for cond, (X, y) in (("normal", (Xs, ys)), ("EDA", (Xa, ya))):
                t0 = time.time()
                acc = train_fn(X, y, Xdv, ydv, Xte, yte, seed, kv)
                rows.append((size, seed, cond, len(X), acc))
                print(f"N={size!s:>5} seed={seed} {cond:>6} (train sents={len(X):>6}) "
                      f"test acc={acc*100:.2f}%  [{time.time()-t0:.0f}s]")

    with open(os.path.join(outdir, "raw_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_train", "seed", "condition", "train_sentences", "test_accuracy"])
        w.writerows(rows)

    # summary table
    print("\n=== SUMMARY (mean +/- std over seeds, test accuracy %) ===")
    print(f"{'N_train':>8} | {'normal':>14} | {'+EDA':>14} | {'gain':>6} | paper CNN avg (5 datasets): normal -> +EDA")
    summary = []
    for size in sizes:
        nm = np.array([r[4] for r in rows if r[0] == size and r[2] == "normal"]) * 100
        ed = np.array([r[4] for r in rows if r[0] == size and r[2] == "EDA"]) * 100
        p = PAPER_CNN_AVG[size]
        summary.append((size, nm.mean(), nm.std(), ed.mean(), ed.std()))
        print(f"{size!s:>8} | {nm.mean():6.2f} ± {nm.std():4.2f} | {ed.mean():6.2f} ± {ed.std():4.2f} "
              f"| {ed.mean()-nm.mean():+5.2f} | {p[0]} -> {p[1]}  ({p[1]-p[0]:+.1f})")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [str(s) if s != "full" else f"full\n({len(Xtr)})" for s in sizes]
    x = np.arange(len(sizes))
    plt.figure(figsize=(6.5, 4))
    plt.errorbar(x, [s[1] for s in summary], [s[2] for s in summary], marker="o", capsize=3, label="Normal")
    plt.errorbar(x, [s[3] for s in summary], [s[4] for s in summary], marker="s", capsize=3, label="With EDA")
    plt.xticks(x, labels)
    plt.xlabel("Training set size (random subset of SST-2)")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"EDA replication on SST-2 ({model.upper()}, {seeds} seeds)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "eda_replication.png"), dpi=200)
    print(f"\nSaved: {outdir}/raw_results.csv and {outdir}/eda_replication.png")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn", choices=["cnn", "lr"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sizes", nargs="+", default=["500", "2000", "5000", "full"])
    a = ap.parse_args()
    sz = [s if s == "full" else int(s) for s in a.sizes]
    run(sizes=sz, seeds=a.seeds, model=a.model)
