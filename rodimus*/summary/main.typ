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
#let AA = $bold(a)$
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


