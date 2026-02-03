#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]
#set math.equation(numbering: "(1)")

#text(size: 2em)[#link(
  "https://openreview.net/pdf?id=IIVYiJ1ggK#page=0.75",
)[Rodimus\*: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions]]

= Abstract
Rodimus is essentially a type of SSM
- Like the input-dependent selection mechanism in Mamba, this paper introduces Data-Dependent Tempered Selection (DDTS), which has the benefit over Mamba of requiring a smaller hidden state expansion factor for the same performance

Rodimus+ combines Rodimus with Sliding Window Shared-Key Attention (SW-SKA), based on the observation that attention usually targets nearby tokens
- We then rely on the SSM-like nature of Rodimum for more long-range information retrieval

This means that both solutions, Rodimus\*, have linear computational cost in the sequence length
- Rodimus is in fact actually a linear approximation of attention

= Introduction
Much work into efficient alternative to attention, with three main categories to compress the KV cache in softmax attention from distinct perspectives:
- Semantic i.e. linear attention, or linear State Space Models (SSMs)
  - Compressing softmax attention's unlimited capacity into linear attention with fixed capacity leads to some information loss, but current methods try to address this by 1) enhancing memory capacity and 2) utilizing the fixed-sized states more effectively e.g. data-dependent decay/gating/filtering/selection, etc, whilst being careful that this does not hinder parallel training efficiency (Mamba does this with its parallel-scan algorithm).
- Token compression where sparsity is introduced into the attention mask to follow predefined patterns and focus on strategically chosen tokens
- Head compression where the design of attention heads is modified
  - While these methods reduce the KV cache size by a constant factor, Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) are lossy compared to Multi-Head Attention (MHA)

= Preliminaries
#let QQ = $bold(Q)$
#let KK = $bold(K)$
#let VV = $bold(V)$
#let XX = $bold(X)$
#let OO = $bold(O)$
#let WW = $bold(W)$
#let MM = $bold(M)$

Regular attention ($MM$ is just the mask matrix):
$
  QQ, KK, VV = XX WW_Q, XX WW_K, XX WW_V \
  OO = "softmax" ((QQ KK^top) dot.o MM) VV
$

This can be equivalently be rewritten as (we are simply normalizing after adding all the value vectors)
#let qq = $bold(q)$
#let kk = $bold(k)$
#let vv = $bold(v)$
#let xx = $bold(x)$
#let oo = $bold(o)$
#let SS = $bold(S)$
#let zz = $bold(z)$
#let AA = $bold(A)$
#let BB = $bold(B)$
#let CC = $bold(C)$
#let uu = $bold(u)$

$
  qq_t, kk_t, vv_t = xx_t WW_Q, xx_t, WW_k, xx_t, WW_v \
  bold(o)_t = (sum_(i=1)^t exp(qq_t kk_i^top)vv_i)/sum_(i=1)^t exp(qq_t kk_i^top)
$

To optimizer efficiency, linear attention mechanisms replace the exponential (also known as Radial Basis Function (RBF) or Gaussian) kernel $exp(qq_t kk_i^top)$ with a kernel $k(qq_t kk_i^top)$ paired with an associated feature map $phi.alt$ i.e. $k(qq_t kk_i^top) = phi.alt(qq_t) phi.alt(kk_i)^T$
- We can't simply use the exponential kernel as its feature map is infinite-dimensional, so we have to use another kernel that can act as a distance i.e. has positive output ($RR^+$)
- With a kernel, we can simply the calculation of  $oo_t$ as:
$
  oo_t = (sum_(i=1)^t phi.alt(qq_t) phi.alt(kk_i)^top vv_i)/ (sum_(i=1)^t phi.alt(qq_t) phi.alt(kk_i)^top) = (phi.alt(qq_t) sum_(i=1)^t phi.alt(kk_i)^top vv_i)/(phi.alt (qq_t) sum_(i=1)^t phi.alt(kk_i)^T)
$
- Letting $SS_t = sum_(i=1)^t phi.alt(kk_i)^top vv_i in RR^(n times m)$ and $zz_t = zz_(t-1) + phi.alt(kk_t)^top in RR^(n times 1)$ be the KV state and K state respectively, we can rewrite the previous equation as a linear State-Space Model (SSM) or RNN, noting that this now has linear complexity as the KV state and K state can be reused for different queries (we have essentially done the key(transposed)-value multiplication first ($cal(O)(d^2 v)$) instead of the query-key(transposed) multiplication ($cal(O)(v^2 d)$))
$
  SS_t = SS_(t-1) + phi.alt(kk_t)^top vv_t \
  zz_t = zz_(t-1) + phi.alt(kk_t)^top \
  oo_t = (phi.alt(qq_t) SS_t)/(phi.alt (qq_t) zz_t)
$
- Try to think of the KV state as a semantic compression of the historical context up to $t$, and when multiplied by $phi.alt(qq_t)$, we can extract the relevant (unnormalized) information out from it
- The denominator $phi.alt(qq_t) zz_t in RR$ may introduced numerical instabilities and hinder the optimization, so recent studies replacing with normalization
- More common to use the identity mapping for $phi.alt$, so the equation becomes:
$
  SS_t = SS_(t-1) + kk_t^top vv_t \
  oo_t = qq_t SS_t
$
- We can also frame these as linear SSMs:
$
  SS_t = AA_t dot.o SS_(t-1) + BB_t dot.o uu_t \
  oo_t = CC_t SS_t
$ <eq:recurrent_expression>
This additive formulation of updating the hidden states $SS_t$ with new key-value pairs at each time step $t$ cannot forget irrelevant information, leading to attention dilution. To address this, we have:
+ Increasing the state size (either $n$ of $d_h$) to retain more information---loses efficiency
+ Incorporate decay factors or gating mechanisms to enable state $SS_t$ to discard irrelevant past information---focus on designing $AA_t$ and $BB_t$ to manage memory retention and forgetting behaviors within the hidden states

Important lessons for design of $AA_t$ and $BB_t$:
- Beneficial for $AA_t$ and $BB_t$ to be negatively correlated e.g. in Mamba{,2} and HGRN2, allowing collaborative regulation of deletion and addition of information
  - Other works e.g. RetNet, GLA, use $AA_t$ as a decay factor, keeping $BB_t$ constant at $1$
- Making $AA_t$ and $BB_t$ to be functions of the input $uu_t$ enable dynamic adjustments over time in a data-dependent manner
- Designs for gating mechanisms must be compatible with GPU acceleration, ensuring that the recurrent expression in @eq:recurrent_expression aligns with a parallel format e.g. parallel scan in Mamba

= Methodology
All existing linear attention models can be expressed recurrently as:
#let aalpha = $bold(alpha)$
#let bbeta = $bold(beta)$
#let ttau = $bold(tau)$
$
  SS_t &= (aalpha_t^top bbeta_t) dot.o SS_(t-1) + (hat(aalpha)_t^top hat(bbeta)_t) dot.o (kk_t^top vv_t) \
  &= (aalpha_t^top bbeta_t) dot.o SS_(t-1) + (hat(aalpha)_t dot.o kk_t)^top (hat(bbeta_t) dot.o vv_t)
$
where $aalpha_t in RR^(1 times n)$ and $bbeta_t in RR^(1 times m)$ denotes the gating mechanism ($n$ is the key dimension and $m$ is the value dimension, with $m >> n$), with the hat versions being negatively correlated, allowing for selection between the new input and the previous state
- By substituting this into @eq:recurrent_expression, we find that linear attention models capture first-order dependencies within sequences in a position-aware manner, so we do not need positional embeddings
- In contrast, the original softmax attention captures only second-order dependencies in a position-agnostic framework, necessitating positional embeddings
- Within the expansion, a $product_(j=(i+1))^t alpha_j$ term regulates absolute positional information, with $aalpha_j dot.o hat(aalpha)_i$ acting as relative positional information akin to RoPE or ALiBi
- Likewise, another $product_(j=(i+1))^t bbeta_j$ term also exists, acting as redundant positional information, but due to its higher $m$ dimensionality, it can impede training efficiency and speed, so Rodimus only focuses on optimizing the design of $aalpha_t$, while forcing $bbeta_t = bold(1)_m$ for all $t$

== Design of $aalpha_t$, $hat(aalpha)_t$, and $hat(bbeta)_t$
$aalpha_t$ is designed such that complete oblivion of the previous state is prevented

$hat(aalpha)_t$ is designed such that although it is negatively correlated with $aalpha_t$, it is asymmetric, introducing greater flexibility into the state transition equation

Additional temperature gate $ttau_t$ governs the sharpness or sensitivity of the selection gate $bold(g)_t$---this temperature helps sharpen the original selection gate $bold(g)_t$, facilitating more aggressive filtering of information

$hat(bbeta)_t$ uses a low-rank formulation to mitigate noise in the input while keeping the number of model parameters manageable

For something to be a valid selection mechanism, as $AA_t$ goes up, $BB_t$ must go down i.e. the product of the gradients of $AA_t$ and $BB_t$ with respect to the input must be negative

== The overall Rodimus block
Computation of control $uu_t = kk_t^top vv_t$ can be interpreted as the state expansion operation within SSMs

Standard Recurrent Neural Networks (and some basic Linear Attention models) process tokens one by one in total isolation before they hit the recurrent state. ShortConv changes this by allowing each timestep $t$ to see a tiny bit of its immediate neighbors (e.g., $t-1$ and $t-2$) before the gating logic is even calculated.
- Local Aggregation: it blends the current input $x_t$ with the preceding few tokens.
- Feature Evolution: instead of $bold(g)_t$ and $ttau_t$ being based purely on a single vector $xx_t$, they are now based on a local window.
- It adds local shift invariance

== Shared-Key Attention for Lossless Head Compression
If we use the same key projection for all heads, the query projections per head can actually compensate for this by turning the single key projection into a key projection for a particular head, as part of the _query_ projection
- Thus, we can save on memory, without sacrificing expressivity
- If we use the same value projection for all the heads, we actually lose expressivity
