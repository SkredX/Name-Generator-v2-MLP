# Theory & Mathematics Behind the Trigram + MLP Interactive Name Generator

This document explains the **logic**, **mathematical functionality**, and **analytical theory** behind an interactive **trigram + MLP** (Multi-Layer Perceptron) based name generator.

We will **not** discuss implementation details or Python syntax. Instead, we focus on the *conceptual* and *mathematical* ideas that the code represents.

---

## 1. Problem Setup: Character-Level Name Modeling

We study a dataset of names, each represented as a sequence of characters:

\[
w = (c_1, c_2, \dots, c_L),
\]

where each \(c_i\) is a character from a finite vocabulary \(\mathcal{V}\) (letters a–z plus a special “dot” symbol `.` used for start/end).

To make the model handle **beginning** and **end** of names explicitly, every word is padded:

\[
\tilde{w} = (., c_1, c_2, \dots, c_L, .),
\]

where `.` is represented as a special token with index 0.

The overall goal is to learn a probabilistic model of the form:

\[
P(c_{t} \mid c_{t-1}, c_{t-2}, \dots),
\]

so that we can **sample** new names by drawing successive characters from this learned conditional distribution until we reach an end token `.`.

---

## 2. Trigram (n-gram) Language Model

### 2.1. N-gram Idea

An **n-gram model** approximates the probability of the next character using only the last \(n-1\) characters.  
For **trigrams** (\(n=3\)):

\[
P(c_t \mid c_1,\dots,c_{t-1}) \approx P(c_t \mid c_{t-2}, c_{t-1}).
\]

Similarly:

- Unigram: \(P(c_t)\) (no context)
- Bigram: \(P(c_t \mid c_{t-1})\)
- Trigram: \(P(c_t \mid c_{t-2}, c_{t-1})\)

We build counts of such patterns from the dataset.

### 2.2. Counting and Context Representation

Each character is mapped to an integer index:  
\[
\text{stoi}: \mathcal{V} \to \{0,1,\dots,V-1\}.
\]

For each word \(\tilde{w} = (x_0, x_1, \dots, x_{L+1})\) in indexed form, we collect counts:

- **Unigram** \((n = 1)\): counts of single tokens  
  \[
  \text{count}(x_t).
  \]
- **Bigram** \((n = 2)\): counts of pairs \((x_{t-1} \to x_t)\)  
  \[
  \text{count}(x_{t-1}, x_t).
  \]
- **Trigram** \((n = 3)\): counts of triples \((x_{t-2}, x_{t-1} \to x_t)\)  
  \[
  \text{count}(x_{t-2}, x_{t-1}, x_t).
  \]

For a given order \(n\), a **context** is the previous \(n-1\) tokens:

\[
\text{context} = (x_{t-(n-1)}, \dots, x_{t-1}).
\]

We maintain a mapping from contexts to **frequency distributions** over next tokens.

### 2.3. Laplace Smoothing

Given counts for a context \(c\) and a vocabulary of size \(V\), we want a probability distribution:

\[
P_{\text{ngram}}(w \mid c) = \frac{\text{count}(c \to w) + \alpha}{\sum_{w'} \big(\text{count}(c \to w') + \alpha\big)},
\]

where \(\alpha > 0\) is the **Laplace smoothing** parameter (here denoted `ALPHA_LAPLACE`).

This ensures:
- No probability is exactly zero (even for unseen transitions).
- The distribution is properly normalized.

### 2.4. Backoff Across N-gram Orders

We maintain **unigram, bigram, and trigram** counts. For a given current context (sequence of observed tokens), we:

1. Try to use the **highest-order** model (trigram).  
2. If that context has never been seen, back off to bigram, then unigram.  
3. If even unigram context is absent (very unlikely), we reduce to essentially a smoothed uniform distribution.

Thus, we choose the **best available n-gram order** \(\hat{n} \in \{1,2,3\}\) and its context \(c\), and a count \(N_c\) (total occurrences of that context in the dataset):

- Order: \(\hat{n}\)
- Context: \(c\)
- Count: \(N_c\)

This yields a **discrete probability vector**:

\[
\mathbf{P}_{\text{ngram}} \in \mathbb{R}^V,
\quad
\sum_{i=1}^V \mathbf{P}_{\text{ngram},i} = 1.
\]

---

## 3. Supervised Dataset for the MLP

### 3.1. Context–Target Pairs

We want the neural network to learn:

\[
f_\theta: \underbrace{(x_{t-3}, x_{t-2}, x_{t-1})}_{\text{BLOCK\_SIZE} = 3}
\mapsto \text{distribution over next token } x_t.
\]

For each padded word \(\tilde{w} = (x_0, x_1, \dots, x_{L+1})\), we create training examples:

- Input context (length 3): \((x_{t-3}, x_{t-2}, x_{t-1})\)
- Target next token: \(x_t\)

where we initially set the context to \((0, 0, 0)\) and slide it across the sequence as we move forward through the word.

This gives us a supervised dataset:
- Inputs: \(X \in \mathbb{N}^{N \times 3}\)
- Targets: \(Y \in \mathbb{N}^{N}\),

where each input row is a context of 3 token indices, and each target is the index of the next character.

### 3.2. Train/Dev/Test Split

The dataset is split into:

- Training set (e.g. 80%),
- Development/validation set (e.g. 10%),
- Test set (e.g. 10%).

This allows us to:
- Train the model on training data,
- Tune hyperparameters/track overfitting via dev loss,
- Optionally evaluate final performance on unseen test data.

---

## 4. Neural Network Model (MLP)

The MLP performs a **nonlinear mapping** from a sequence of token indices to a distribution over next tokens.

### 4.1. Embedding Layer

Each token index \(i \in \{0,\dots,V-1\}\) is mapped to a continuous vector:

\[
\mathbf{e}_i \in \mathbb{R}^{d},
\]

where \(d = \text{EMBED\_DIM}\).

A learned embedding matrix:

\[
E \in \mathbb{R}^{V \times d}
\]

holds all embedding vectors as rows. When given a context \((x_{t-3}, x_{t-2}, x_{t-1})\), the embedding layer returns:

\[
\left[
\mathbf{e}_{x_{t-3}},
\mathbf{e}_{x_{t-2}},
\mathbf{e}_{x_{t-1}}
\right] \in \mathbb{R}^{3 \times d}.
\]

These embeddings are **learned parameters**, capturing similarity between characters in a continuous space.

### 4.2. Flattening Step

The embedding outputs for the block of size 3 are flattened into a single vector:

\[
\mathbf{z} = \text{vec}\left(
\begin{bmatrix}
\mathbf{e}_{x_{t-3}} \\
\mathbf{e}_{x_{t-2}} \\
\mathbf{e}_{x_{t-1}}
\end{bmatrix}
\right) \in \mathbb{R}^{3d}.
\]

This becomes the input to the hidden layer.

### 4.3. Hidden Layer with Nonlinearity

A fully-connected layer with weights \(W_1 \in \mathbb{R}^{3d \times h}\) and bias \(\mathbf{b}_1 \in \mathbb{R}^{h}\) computes:

\[
\mathbf{h} = \sigma( \mathbf{z} W_1 + \mathbf{b}_1 ),
\]

where:
- \(h = \text{HIDDEN\_DIM}\),
- \(\sigma\) is a **nonlinear activation**, e.g. ReLU:
  \[
  \sigma(x) = \max(0, x)
  \]
  applied elementwise.

This transformation allows the model to learn **nonlinear interactions** between characters in the context.

### 4.4. Output Layer and Logits

A second linear layer with weights \(W_2 \in \mathbb{R}^{h \times V}\) and bias \(\mathbf{b}_2 \in \mathbb{R}^{V}\) produces **logits**:

\[
\mathbf{o} = \mathbf{h} W_2 + \mathbf{b}_2 \in \mathbb{R}^{V}.
\]

These logits are unnormalized scores for each possible next token.

### 4.5. Softmax and Predicted Distribution

The model’s predicted probability distribution over the next token is obtained by applying the **softmax** function:

\[
P_{\text{MLP}}(w \mid x_{t-3}, x_{t-2}, x_{t-1}) = 
\frac{\exp(o_w)}{\sum_{j=1}^{V} \exp(o_j)}.
\]

This yields:

\[
\mathbf{P}_{\text{MLP}} \in \mathbb{R}^V,
\quad
\sum_{i=1}^{V} \mathbf{P}_{\text{MLP},i} = 1.
\]

---

## 5. Training Objective and Optimization

### 5.1. Cross-Entropy Loss

Given a context (input) and a true next-token \(y\), the model outputs \(\mathbf{P}_{\text{MLP}}\). The **cross-entropy loss** is:

\[
\mathcal{L} = -\log P_{\text{MLP}}(y \mid \text{context}) 
= - \log \mathbf{P}_{\text{MLP},y}.
\]

For a minibatch of size \(B\), the loss is averaged:

\[
\mathcal{L}_{\text{batch}} = -\frac{1}{B} \sum_{i=1}^{B} \log \mathbf{P}_{\text{MLP},y_i}.
\]

This is equivalent to **maximizing the log-likelihood** of the true next characters given their contexts.

### 5.2. Gradient-Based Optimization (Adam)

The model parameters \(\theta\) consist of:

\[
\theta = \{E, W_1, \mathbf{b}_1, W_2, \mathbf{b}_2\}.
\]

We use an optimizer (e.g., Adam) to perform **stochastic gradient descent** on the loss:

\[
\theta \leftarrow \theta - \eta \, \hat{\nabla}_\theta \mathcal{L},
\]

where:

- \(\hat{\nabla}_\theta \mathcal{L}\) is the gradient estimate over a minibatch,
- \(\eta\) is the learning rate (`LR`).

Adam adaptively adjusts effective learning rates per parameter using estimates of first and second moments of gradients, giving more robust convergence properties.

### 5.3. Epochs, Dev Loss, and Generalization

- Training runs for several **epochs**, each consisting of multiple minibatch updates.
- After each epoch, we compute **dev loss** on held-out data.
- A decreasing train loss with stable or slightly higher dev loss indicates learning; rising dev loss vs train loss suggests overfitting.

---

## 6. Interpolating Trigram and MLP Distributions

The key idea of this model is to **combine the strengths** of:

- The **n-gram** model (good for frequent, well-observed contexts),
- The **MLP** model (can generalize to contexts never seen exactly before).

### 6.1. Confidence-Based Weighting

For the current context (sequence of tokens observed so far), we:

1. Choose the best available n-gram order \(\hat{n}\) and its context \(c\).
2. Obtain its total context count \(N_c\), i.e., how many times we’ve seen \(c\) followed by any next token in the training data.
3. Compute an **interpolation weight**:

\[
w = \frac{N_c}{N_c + K},
\]

where \(K = \text{K\_CONFIDENCE\_K}\) is a tunable parameter.

Interpretation:

- If \(N_c\) is large (context seen many times), \(w \approx 1\), so we trust the **n-gram model** more.
- If \(N_c\) is small (context rarely seen), \(w \approx 0\), so we lean more on the **MLP model**.

### 6.2. Mixture Distribution

Let:

- \(\mathbf{P}_{\text{ngram}} \in \mathbb{R}^V\) be the n-gram distribution for the chosen order \(\hat{n}\) and context \(c\),
- \(\mathbf{P}_{\text{MLP}} \in \mathbb{R}^V\) be the neural network’s distribution for the **same** context (prepared appropriately as a block of last 3 tokens).

The **combined** or **final** distribution is:

\[
\mathbf{P}_{\text{final}} 
= w \cdot \mathbf{P}_{\text{ngram}} 
+ (1 - w) \cdot \mathbf{P}_{\text{MLP}}.
\]

This vector is then renormalized to ensure it sums to 1 (numerical stability). Conceptually, this is a **linear mixture of experts**:

- Expert 1: n-gram model
- Expert 2: MLP model

The mixture coefficient \(w\) depends on **how much data** supports the n-gram context.

---

## 7. Sampling / Name Generation Process

Once the combined model is trained, we generate names by **sampling** characters one by one.

### 7.1. Initial Context

Generation begins with a **start dot** token:

\[
\text{sequence} = [0]
\]

and optionally some user-provided prefix characters mapped to token indices and appended.

### 7.2. Repeated Sampling

At each step:

1. Consider the tokens generated so far:
   \[
   (x_0, x_1, \dots, x_t).
   \]

2. Treat this as the **current context**, and compute:
   - The n-gram distribution \(\mathbf{P}_{\text{ngram}}\),
   - The MLP distribution \(\mathbf{P}_{\text{MLP}}\),
   - The mixture weight \(w\),
   - The final distribution \(\mathbf{P}_{\text{final}}\).

3. Sample the next token \(x_{t+1}\) from \(\mathbf{P}_{\text{final}}\) as a **categorical random variable**:
   \[
   \Pr(x_{t+1} = i) = \mathbf{P}_{\text{final},i}.
   \]

4. Append this new token to the sequence.
5. Stop if:
   - The sampled token is the dot token (end-of-name), or
   - We reach a maximum name length (to avoid infinite loops in degenerate cases).

The resulting sequence of characters (excluding the initial dot, and stopping at the final dot) forms a **generated name**.

---

## 8. Temperature Scaling

To control the **randomness** of generation, **temperature** \(T > 0\) is applied to logits or probabilities.

Given probabilities \(\mathbf{P}_{\text{final}}\), we can define modified logits:

\[
\ell_i = \log(\mathbf{P}_{\text{final},i} + \varepsilon)
\]

and scale them:

\[
\tilde{\ell}_i = \frac{\ell_i}{T}.
\]

Then new probabilities are:

\[
\tilde{P}_i = \frac{\exp(\tilde{\ell}_i)}{\sum_j \exp(\tilde{\ell}_j)}.
\]

Effects of \(T\):

- \(T < 1\): distribution becomes **sharper**, more **greedy**, less diverse names.
- \(T > 1\): distribution becomes **flatter**, more **random**, more creative but possibly less coherent.

Thus, temperature offers a **knob** between **safe/predictable** and **creative/chaotic** sampling behavior.

---

## 9. Top-k Sampling

Top-k sampling further controls randomness and quality.

Given a probability vector \(\mathbf{P}\), we:

1. Identify the \(k\) indices with the highest probabilities.
2. Zero out probabilities for all other tokens.
3. Renormalize the remaining probabilities to sum to 1.
4. Sample only from this restricted set.

Mathematically:

- Let \(S\) be the set of indices of the top-k probabilities.
- Define:
  \[
  \hat{P}_i =
  \begin{cases}
  \mathbf{P}_i & \text{if } i \in S, \\
  0 & \text{otherwise}.
  \end{cases}
  \]
- Renormalize:
  \[
  \tilde{P}_i = \frac{\hat{P}_i}{\sum_j \hat{P}_j}.
  \]

Sampling from \(\tilde{\mathbf{P}}\) keeps us within the **most plausible** characters while still preserving some randomness.

---

## 10. Analytical Summary

1. The **trigram (n-gram) component** is a **count-based model** that captures local character patterns directly from the data with smoothing. It is highly **data-driven**, strong where counts are high, but weak for unseen contexts.

2. The **MLP component** learns a **continuous representation** of characters (through embeddings) and a nonlinear mapping from context to next-character distribution. It can **generalize** across contexts that are similar but not identical, thanks to shared parameters and embeddings.

3. The **combined model** is a **mixture of experts**:
   \[
   \mathbf{P}_{\text{final}} = w \cdot \mathbf{P}_{\text{ngram}} + (1-w) \cdot \mathbf{P}_{\text{MLP}},
   \]
   where \(w\) depends on the **context frequency** \(N_c\). This makes the system **adaptive**:
   - In high-data regions, rely more on empirical counts.
   - In low-data regions, rely more on learned neural generalization.

4. **Temperature** and **top-k** sampling provide **controllable randomness**, enabling a spectrum from deterministic to highly creative behavior.

5. The entire framework defines an explicit **probability distribution** over sequences of characters, letting us do **both**:
   - **Training** by maximizing log-likelihood (minimizing cross-entropy),
   - **Generation** by sampling from the learned distribution.

This combination of discrete statistics and neural modeling leads to a robust, interpretable, and flexible interactive **name generator**.
