#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://arxiv.org/pdf/2502.09992")[Large Language Diffusion Models]]

= Abstract
Forward data masking and reverse generation (mask predictor) process parameterized by a Transformer to predict masked tokens.

= Introduction
Paper argues the scalability is primarily a consequence of interplay between Transformers, model size, data size and Fisher consistency---the ability to recover the true data distribution with infinite data, a sufficiently large network, and optimal training.
- Moreover instruction-following and in-context learning are intrinsic properties of all conditional generative models on structurally consistent linguistic tasks.

Left-to-right generation restricts ARMs' ability to handle reverse reasoning tasks.
- In contrast, llada can model bidirectional dependencies.

Llada seems to show better instruction-following capabilities after Supervised Fine Tuning (SFT).

== Approach
Forward process masks tokens independently and runs from $t in (0, 1)$, with each token being masked with probability $t$, or remaining unmasked with probability $1-t$.

Reverse process iteratively predicts masked tokens as $t$ moves from $1 -> 0$.

The mask predictor $p_theta (dot|x_t)$ predicts all masked tokens $bold(M)$ simultaneously.

Cross-entropy loss that is computed only on masked tokens is:
$ cal(L) (theta) eq.delta -bb(E)_t,x_theta,x_t [1/t sum_(i=1)^L bold(1) [x_t^i = bold(M)] log p_theta (x_0^i|x_t)] $
where $x_0$ is a training sample, $t ~ cal(U) [0,1]$ $x_t$ sampled from the forward process, and $L$ sequence length.

Both the forward and reverse processes can be viewed as discretizations of DDPM, and thus dealing with the fact that unlike the continuous case where we can partially remove information from say an image, we have to either completely mask, or unmask tokens.
- The unmasking of the model, where some tokens can be remasked kind of seems like the model predicting the entire noise $epsilon_theta$, but we only remove some multiple of it in the actual sampling stage.

== Pre-training
Llada does not use a causal mask, unlike existing LLMs.

Vanilla multi-head attention, instead of grouped query attention used, as Llada is incompatible with KV caching.
- Attention layer has more paramteres, so FFN dimension reduced for comparable model size.

For a training sequence $x_0$, randomly sample $t in [0,1]$, and mask each token independently with the same probability $t$ to obtain $x_t$.

To handle random lengths, 1T of pre-training data sampled to a random length $in [1, 4096]$.

Warmup-Stable-Decay LR schedulin used with AdamW optimizer.

== Supervised Fine-Tuning
Really the same as pre-training, apart from the fact that we now have paired data of prompt and response $(p_0, r_0)$, with all the masked tokens in the response. The input of the model is simply a concatenation of the two.

EOS tokens appended to ensure equal lengths across all data. It is treated as a normal token during training, and removed during sampling, so that Llada can control the response length automatically.

== Inference
Generation length is treated as a hyperparameter, with tokens after the EOS token being discarded.

An intermediate sampling step from time $t in (0,1]$ to $s in [0,t)$ consists of the following:
- Feed $p_0$ and $r_t$ in to the mask predictor and predict all masked tokens simultaneously.
- Remask $s/t$ of the predicted tokens in expectation to obtain $r_s$. This ensures the transition of the reverse process aligns with the forward process for accurate sampling.
- Although the remasking strategy should be purely random, inspired by annealing tricks of sampling in LLMs, a low-confidence remasking strategy is used, where $s/t$ of predicted tokens with the lowest confidence are remasked.

Llada enables flexible sampling, supporting autoregressive and block diffusion, but diffusion sampling yields the best performance.

For conditional likelihood evaluation, while the loss above could be utilized, the following equivalent form exibits lowest variance and is more stable:
$ -bb(E)_l,r_theta,r_l [L/l sum_(i=1)^L bold(1) [r_l^i = bold(M)] log p_theta (r_0^i|p_0,r_l)] $
where $L$ is the sequence length of $r_0$, $l$ is uniformly sampled from ${1,2,...,L}$, and $r_l$ is obtained by uniformly sampling $l$ tokens from $r_0$ without replacement for masking.

= Experiments

