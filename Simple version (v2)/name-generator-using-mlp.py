# imports, seed and config
import os
import glob
import math
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DATA=["/kaggle/input/names-dataset/names.txt"]
MAX_NGRAM = 3          # build unigram, bigram, trigram counts
BLOCK_SIZE = 3         # context length fed to MLP (number of previous tokens)
EMBED_DIM = 24         # embedding dim for tokens in MLP
HIDDEN_DIM = 128       # hidden units in MLP
BATCH_SIZE = 512
EPOCHS = 12            # adjust for runtime / quality tradeoff
LR = 1e-3
ALPHA_LAPLACE = 1.0    # Laplace smoothing alpha
K_CONFIDENCE_K = 5.0   # interpolation hyperparameter
MAX_NAME_LEN = 30
PRINT_EVERY = 200
SEED_FOR_SAMPLING = 2147483647

# dataset loading
def data_path(candidates):
    for p in candidates:
        if "*" in p:
            found = glob.glob(p)
            if found:
                return found[0]
        if os.path.exists(p):
            return p
    return None

DATA_PATH = data_path(DATA)
if DATA_PATH is None:
    raise FileNotFoundError("names.txt not found. Upload it to working dir or adjust DATA.")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_words = [line.strip() for line in f if line.strip()]
words = [w.lower() for w in raw_words]
print(f"Loaded {len(words)} names. Example: {words[:8]}")

# vocab
chars = sorted(list(set("".join(words))))
chars = [c for c in chars if c.isalpha()]  # keep alphabetic only
itos = {0: "."}  # dot token for start/end
for i, ch in enumerate(chars, start=1):
    itos[i] = ch
stoi = {s: i for i, s in itos.items()}
V = len(itos)
print("Vocab size:", V)
print("Some tokens:", dict(list(itos.items())[:10]))

# ngrams counters
def build_ngram_counters(corpus: List[str], max_n: int) -> Dict[int, Dict[Tuple[int, ...], Counter]]:
    counters = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    for w in corpus:
        seq = [0] + [stoi[c] for c in w] + [0]
        for n in range(1, max_n + 1):
            ctx_len = n - 1
            for i in range(len(seq) - ctx_len):
                ctx = tuple(seq[i:i+ctx_len]) if ctx_len > 0 else tuple()
                nxt = seq[i+ctx_len]
                counters[n][ctx][nxt] += 1
    return counters

ngram_counters = build_ngram_counters(words, MAX_NGRAM)
for n in range(1, MAX_NGRAM+1):
    print(f"Order {n} contexts: {len(ngram_counters[n])}")

def ngram_probs_for_context(counters, n, context, alpha, vocab_size):
    cnts = counters[n].get(context, None)
    if cnts is None:
        arr = np.ones(vocab_size, dtype=float) * alpha
        arr = arr / arr.sum()
        return arr
    arr = np.array([cnts.get(i, 0) for i in range(vocab_size)], dtype=float)
    arr += alpha
    arr = arr / arr.sum()
    return arr

# building supervised dataset for multilevel perceptron
def build_supervised_dataset(corpus: List[str], block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X, Y = [], []
    for w in corpus:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(list(context))
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

random.shuffle(words)
n = len(words)
n1 = int(0.8 * n)
n2 = int(0.9 * n)
Xtr, Ytr = build_supervised_dataset(words[:n1], BLOCK_SIZE)
Xdev, Ydev = build_supervised_dataset(words[n1:n2], BLOCK_SIZE)
Xte, Yte = build_supervised_dataset(words[n2:], BLOCK_SIZE)
print("Shapes Xtr, Xdev, Xte:", Xtr.shape, Xdev.shape, Xte.shape)

# mlp and training loop
class SimpleMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.block_size = block_size
        self.fc1 = nn.Linear(embed_dim * block_size, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        e = self.embed(x)               # (batch, block_size, embed_dim)
        e = e.view(e.size(0), -1)       # (batch, block_size*embed_dim)
        h = self.act(self.fc1(e))
        logits = self.fc2(h)
        return logits

model = SimpleMLP(V, EMBED_DIM, BLOCK_SIZE, HIDDEN_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

#training
def get_random_batch(X, Y, batch_size):
    idx = torch.randint(0, X.shape[0], (batch_size,))
    return X[idx], Y[idx]

def train(model, Xtr, Ytr, Xdev, Ydev, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model.train()
    iters = max(1, Xtr.shape[0] // batch_size)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for it in range(iters):
            xb, yb = get_random_batch(Xtr, Ytr, batch_size)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (it + 1) % PRINT_EVERY == 0:
                print(f"Epoch {epoch+1} iter {it+1}/{iters} loss {loss.item():.4f}")
        avg_loss = epoch_loss / iters
        model.eval()
        with torch.no_grad():
            dev_logits = model(Xdev)
            dev_loss = criterion(dev_logits, Ydev).item()
        model.train()
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}, dev_loss={dev_loss:.4f}")
    return model

print("Training MLP (this may take time)...")
model = train(model, Xtr, Ytr, Xdev, Ydev)
print("Training finished.")

# combining ngram and mlp distributions
def get_best_ngram_order_and_context(context_tokens: List[int], counters):
    for order in range(MAX_NGRAM, 0, -1):
        ctx_len = order - 1
        if ctx_len == 0:
            ctx = tuple()
        else:
            if len(context_tokens) < ctx_len:
                continue
            ctx = tuple(context_tokens[-ctx_len:])
        cnt_map = counters[order].get(ctx, None)
        if cnt_map:
            total = sum(cnt_map.values())
            return order, ctx, total
    return 1, tuple(), sum(ngram_counters[1].get(tuple(), Counter()).values())

def compute_combined_distribution(context_tokens: List[int]):
    order, ctx, ctx_count = get_best_ngram_order_and_context(context_tokens, ngram_counters)
    P_ngram = ngram_probs_for_context(ngram_counters, order, ctx, ALPHA_LAPLACE, V)
    # Prepare block_size context for MLP input (pad left with zeros if necessary)
    ctx_for_mlp = [0] * max(0, BLOCK_SIZE - len(context_tokens)) + context_tokens[-BLOCK_SIZE:]
    x = torch.tensor([ctx_for_mlp], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(x).squeeze(0)
        probs_mlp = F.softmax(logits, dim=0).cpu().numpy()
    weight = ctx_count / (ctx_count + K_CONFIDENCE_K)
    P_final = weight * P_ngram + (1.0 - weight) * probs_mlp
    P_final = np.maximum(P_final, 1e-12)
    P_final = P_final / P_final.sum()
    return P_final, order, ctx, ctx_count, weight

# sampling utilities

def sample_name(prefix: str = "", max_len: int = MAX_NAME_LEN, seed: int = None,
                temperature: float = 1.0, top_k: int = None):

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    prefix_chars = [c for c in prefix.lower() if c.isalpha()]
    for c in prefix_chars:
        if c not in stoi:
            raise ValueError(f"Character '{c}' not in vocabulary")

    seq = [0] + [stoi[c] for c in prefix_chars]  # include starting dot then prefix tokens
    generated = []
    for _ in range(max_len):
        P_final, order, ctx, cnt, weight = compute_combined_distribution(seq)
        # apply temperature
        if temperature != 1.0:
            logits = np.log(P_final + 1e-20) / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
        else:
            probs = P_final
        # top-k filter
        if top_k is not None and 0 < top_k < V:
            top_idxs = np.argpartition(-probs, min(top_k, V)-1)[:min(top_k, V)]
            mask = np.zeros_like(probs, dtype=bool)
            mask[top_idxs] = True
            filtered = probs * mask
            if filtered.sum() <= 0:
                filtered = probs
            probs = filtered / filtered.sum()
        nxt = rng.choice(np.arange(V), p=probs)
        if nxt == 0:
            break
        generated.append(itos[nxt])
        seq.append(nxt)
        if len(generated) >= max_len:
            break

    prefix_str = "".join(prefix_chars)
    final = (prefix_str + "".join(generated)).capitalize()
    return final

def interactive_loop():
    print("N-gram + MLP Name Generator (interactive). Do note that Temp scales randomness so low temp is mild+safe and high temp is creative+chaotic. top-k limits choices to top k characters")
    while True:
        try:
            prefix = input("Enter starting letters (leave empty for any): ").strip()
        except Exception:
            print("input() not available in this environment. Exiting interactive mode.")
            break
        if prefix.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        k_str = input("How many names to generate? [default 10]: ").strip()
        try:
            k = int(k_str) if k_str else 10
            if k <= 0:
                raise ValueError()
        except:
            k = 10

        temp_str = input("Temperature (float, default 1.0): ").strip()
        try:
            temperature = float(temp_str) if temp_str else 1.0
        except:
            temperature = 1.0

        topk_str = input("Top-k (int, optional): ").strip()
        try:
            top_k = int(topk_str) if topk_str else None
        except:
            top_k = None

        seed_str = input("Seed (int, optional): ").strip()
        try:
            seed = int(seed_str) if seed_str else None
        except:
            seed = None

        results = []
        attempts = 0
        while len(results) < k and attempts < 10 * k:
            nm = sample_name(prefix=prefix, max_len=MAX_NAME_LEN, seed=(seed + attempts) if seed is not None else None,
                             temperature=temperature, top_k=top_k)
            if nm and nm not in results:
                results.append(nm)
            attempts += 1

        print(f"\nGenerated {len(results)} names (prefix='{prefix}'):")
        for i, nm in enumerate(results, 1):
            print(f"{i:2d}. {nm}")
        print()

        cont = input("Continue? (y/n) [default y]: ").strip().lower()
        if cont in ("n", "no"):
            print("Stopping interactive session.")
            break

interactive_loop()
