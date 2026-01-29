#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link(
  "https://arxiv.org/pdf/2410.04466",
)[LLM Inference Acceleration]]

= Abstract
Three edge system trends:
- Multi-modality
- Inference-time compute
- Inference energy efficiency

= Introduction
Optimizations include quantization, sparsity, and fast decoding for generative LLMs

= Generative LLM architecture
Flash attention uses blockwise softmax computation, storing additional statistics of the running max, and running sum so that the intermediate $n times n$ attention score matrix is not materialized in HBM, with only the final output with accumulated values being materialized in HBM
- It also uses gradient checkpointing in the backwards pass so that only the output and previously mentioned softmax statistics are stored, and we can recompute the attention scores in blocks again all in SRAM
- Flash attention 2 is essentially a hardware-level improvement, with no real change to the algorithm itself

== Attention-based LLM
Transformer-XL uses segment-level recurrence to learn dependencies beyond a fixed length without disrupting temporal coherence

Linear transformer uses a linear dot product of kernel feature map representation of self-attention
- Allows it to change the computation order, resulting in linear complexity in length

Attention-Free Transformer (AFT) combines key and value with positional biases, then performs element-wise multiplication with the query
- Also linear memory complexity
- Based on AFT, we also have Receptance Weighted Key Value (RWKV) combining efficient parallel transformer training with efficient RNN inference. It uses linear attention, and allows the model to be expressed as either a transformer or an RNN

#link("https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2403.19928&hl=en&sa=T&oi=gsr-r&ct=res&cd=0&d=9451344766463580985&ei=lXN2aaTsOpi8ieoPv_jKaQ&scisig=AHkA5jQLKGJhChinp3KDwB7v_5aM")[This paper] shows that improved attention mechanisms require expensive retraining, impractical for LLMs
- Kernalization method based on Discrete Cosine Transform (DCT) to convert standard Transformer into a model with linear complexity and low training costs
- Low training costs through a weighted quasi-Monte Carlo method for sampling

== SSM-based LLM
State Space Models (SSMs) refer to simple linear models of the form:
$
  h & = A times h + b times x \
  y & = C times h
$
- These equations actually come from ODEs
- SSMs are like RNNs where the dynamics are constrained for efficiency and long memory

Where does the SSM originate from?
- Notation: $x(t)$ refers to the hidden state/memory at continuous time $t$, and $u(t)$ is the input
- The memory evolves continuously through time---think about it as the position of the mass in a spring-mass system
- $dot(x)(t)$ tells you how fast, and in which direction memory changes
- Memory then is simply the integral of all previous changes:
$
  x(t) = integral_0^t dot(x)(tau) dif tau + x(0)
$
- Thus the SSM equation is $dot(x) (t) = A x(t) + B u(t)$, and *not* $x(t) = f(x(t), u(t))$ as that would be a static map, forgoing this notion of memory accumulating over time
- Also from classical control/signal processing, linear systems, which have guaranteed solutions in the form of exponentials, are described as: $"rate of change of state" = A x + B u$
- Solutions are of the form $x(t) = e^(A t) x(0) + integral_0^t e^(A (t - tau)) B u(tau) dif tau$
  - Each past input decays smoothly according to the eigenvalues of $A$, with the real part of each eigenvalue controlling the timescale

Structured State Space Model (S4) conditions the matrix $A$ with low-rank corrections, enabling it to be stably diagnoalized (and so we can easily compute intermediate hidden states), simplifying the SSM to computations that involve in-depth exploration of the Cauchy kernel
- Higher computational efficiency, while retaining theoretical advantages

Gated State Space Model (GSS) uses gated activation functions to get faster training speeds compared to S4

Hyena addresses that sub-quadratic methods on low-rank and sparse approximations need to be combined with dense attention layers to match Transformers
- So attention needs to be avoided entirely?
- Dense attention captures arbitrarily long-range interactions, and fine-grained token-token dependencies
- Instead of asking how much should token $t$ attend to token $s$, it's what transformations should signals from distance $t - s$ undergo before reaching $t$
- The Hyena assumption is that most useful token interactions are not arbitrary $L^2$ patterns
- The key thing with Hyena/Mamba/etc. is that unlike RNNs like LSTM/GRU, they are *effectively* stateless across time, so the hidden states are functions of previous outputs, not previous hidden states
- Theoretically something like an LSTM can learn to act like say Mamba, but Hyena/Mamba bake a certain kernel structure (the filters that control information propagation) into the model

Mamba makes the SSM parameters a function of the input and enables the model to selectively propagate or forget information along the sequence length dimension based on the current token

DenseSSM enhances Mamba by selectively integrating shallow layer hidden states into deeper layers.

Mamba2 also introduces the State Space Duality (SSD) framework

== Hybrid LLM
Attention used for information extraction and SSM used for information compression

Block-State-Transformer (BST) uses transformer within blocks and SSM for long-range contextualization
- Also a model called Griffin that does something similar, but with RNN instead of SSM

Jamba interleaves transformer and Mamba layers, with MoE

Infini-Transformer combines masked local attention and long-term linear attention within a single Transformer block

MEGALODON aims to have infinite context length

= Optimizations on Hardware Platforms
Optimizations include:
- Quantization
- Sparsity
- Fast decoding
- Operator optimization
- Heterogeneous cooperation
- Homogeneous cooperation

== Quantization
Different granularities of quantization: group-wise, channel-wise and tensor-wise (more granular)

Uniform and non-uniform quantization---whether or not the range of values are partitioned into equal sized intervals
- Uniform has high computational efficiency, but may fail to capture the data distribution characteristics in uneven data distributions, leading to significant information loss

Weight-only and weight-activation quantization
- Matrix decomposition is another weight-only method to approximate a large matrix with the product of several smaller matrices that can be stored and processed in lower precision formats
- Weight-only mainly reduces model storage requirement, whereas weight-activation can improve inference speed
- Dynamic range quantization important in weight-activation to balance model performance and computational efficiency

For KV cache, there is a pattern than keys are distributed by channel and the values by token, #link("https://arxiv.org/pdf/2505.10938")[other than a small subset that can be excluded from the quantization process as outlier tokens]
- There is also work on #link("https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3600006.3613165&hl=en&sa=T&oi=gsr-r&ct=res&cd=0&d=7491415560688410174&ei=16N3aaagDMyQieoPxcX9yAo&scisig=AHkA5jQWA4tNBQ7cOuoh4pcyWr19")[optimizing hardware scheduling when fetching the KV cache]

== CPU
Methods on CPU mainly target weight-only

Various quantization recipes such as GPTQ, AWQ, and TEQ

To circumvent overheads of weight dequantization from integer to floating on edge CPUs, lookup tables (LUTs) used in T-MAC---also don't need mixed precision matrix multiplication

NoMAD-Attention replaces expensive Multiply-Add (MAD) operations with in-register lookups

DECA designs a near-core ML-model decompression accelerator

== GPU
=== Weight-only
GPTQ is a one-shot weight quantization method using approximate second-order information and error compensation

AWQ protects 1% of salient weights whose activations are extremely large to greatly reduce quantization error

SpQR identifies and isolates outlier-weights, storing them in higher precision

SqueezeLM proposes a sensitivity-based non-uniform quantization method, searching for the optimal bit precision assignment, using second-order information

LLM-MQ also uses sensitivity-based precision (only first-order information), and develops an efficient CUDA kernel by fusing dequantization and general matrix-vector multiplication (GEMV)

APTQ proposes attention-aware quantization, taking into account the nonlinear effect of attention outputs on the entire model

FP6-LLM uses Tensor Core support of floating-point weights for various quantization bit-widths with TC-FPx

Group Quantization and Sparse Acceleration (GQSA) integrations quantization and sparsification in a tightly coupled manner

=== Weight-activation
Can significantly improve performance during pre-fill stage

Compared to weight-only quantization, weight-activation quantization can utilize INT4, INT8, and FP8 computations in special INT tensor cores
- In weight-only, one operand is floating-point, and so the output depends on floating-point activations whose scale is not fixed or known at compile time
- We cannot accumulate the output of integer-float computations into integer, without first converting the float activations to integer (i.e. weight-activation quantization), or alternatively dequantizing the weights to floating point, and then getting the next set of floating point activations (which is what weight-only has to do)

We can hide the weight-activation's methods float-to-int activation quantization overhead with the fact that INT tensor core speedups are much greater, and that the quantization kernels are fused, and hidden by memory latency

Just a note that with floating point formats, we have dynamic scale/range per value, but with integer formats, a block of values shares the same scale/range

LLM.int8  uses vector-wise quantization with separate normalization constants for each inner product in the matrix multiplication, to quantize most of the features. For the outliers, it isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9% of values are multiplied in 8-bit.

SmoothQuant enables INT8 quantization for both weights and activations
- Weights easier to quantize than activations
- Smooths activations outliers by offline migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation

QUIK casts most weights and activations to 4 bits, avoiding outliers, for the prefill phase

Floating point quantization can better handle long-tail or bell-shaped distributions

LLM-FP4 already looks at the model performance aspect of FP4 quantization, and there is negligible accuracy loss, however may actually be slower in GPUs compared to FP16 due to a lack of FP4 hardware

Weight-activation quantization can also reduce total power consumption (compared to weight-only quantization) as additional dequantization operations are not required

= Sparsity
Weight, activation, and attention sparsity

Pruning methods including global pruning, layer-wise pruning, and structured pruning used to achieve weight sparsity
- Reduce the size of weight matrices and use sparse matrix libraries

Activation pruning e.g. threshold pruning and dynamic sparsity used for activation sparsity
- Reduce the computation of activation values
- Hardware optimizations utilize sparse data structures

Attention sparsity involves local attention, block-wise attention, and sparse attention matrices
- Better inference efficiency
- Useful for large-scale data and complex tasks

Random and structured sparsity
- Random sparsity involves a random distribution of zero elements within the matrix---higher accuracy with regards to the original output, but lower computational speed
- Structured patterns e.g. block-wise, N:M (M non-zero elements out of every N elements), channel-wise (prune entire channels in a matrix), and some combinations of these, align with hardware optimizations, improving computational efficiency

== CPU
=== Activation sparsity
Turbo Sparse proposes dReLU activation function

ProSparse also substitutes the activation function, but also includes progressive sparsity regularization, and activation threshold shifting

=== Weight sparsity
SparAMX uses Advanced Matrix Extensions (AMX) to leverage unstructured sparsity

== GPU
Dedicated N:M hardware in modern Nvidia GPUs

=== Activation sparsity
Soft activation sparsity and Low-rank Decomposition (SoLA) retains minority of components, while compressing majority through low-rank decomposition, based on FFN activation patterns

R-Sparse finds two key observations of FFNs, and replaces the linear layers:
+ non-sparse components of input can be regarded as a few bias terms
+ Full computation can be approximated by combination of input channels and weight singular values

=== Weight sparsity
LLM-pruner uses structural pruning to remove non-critical coupled structures based on gradient information
- Then followed by tuning techniques, LoRA, to recover performance

LLMs can be pruned to at least 50% sparsity, without retraining without accuracy loss
- SparseGPT has a sophisticated, iterative pruning process---can even be used during pre-training to pretrain a high sparsity LLM
- Wanda prunes weights with the smallest magnitudes multiplied by the corresponding input activations on a per-output basis---no retraining, or weight update required
- These two also generalize to semi-structured N:M patterns
- E-Sparse introduces entropy to quantify the information richness and also supports N:M

A key bottleneck of LLM inference is skinny matrix multiplications (where at least one dimension is very narrow e.g. batch-1. This makes it harder to amortize the cost of loading the weights
- Even if we use SparseMM, we don't get a very large performance benefit
- Instead, we can try to maximise the Tensor Core utilization via a Load-as-Sparse and Compute-as-Dense methodology for unstructured sparse matrix multiplication
- Flash-LLM proposes a new format called Tiled-CSL to do this

DejaVu predictions contextutal sparsity on the fly given inputs to each layer, avoiding retraining and preserving the LLM's in-context learning ability

Efficient unstructured prunng for Mamba models
- Gradient aware magnitude pruning
- Iterative pruning schedule
- Global pruning strategy

=== Attention sparsity
Very useful during prefilling phase

Static sparsity:
- Sparse Transformer
- StreamingLLM
- Bigbird
- Longformer
- All use a manual combination of global and local patterns to replace full attention patterns

Dynamic sparsity:
- Adaptively sparse attention replaces softmax with $alpha$-entmax, where low-scoring words receive precisely zero weight and drops parts of the context that are no longer required for future generation
- Reformer replaces dot-product attention by using locality-sensitive hashing, reducing complexity from $cal(O) (L^2)$ to $cal(O) (L log L)$
- Sparse Flash Attention extends FlashAttention with key/query dropping and hashing-based attention
- Sparse Sinkhorn Attention uses a learned sorting network to align keys with their relevant query buckets
- $"H"_2"O"$ observes a small portion of tokens contribute most to attention scores, and uses that to guide KV cache eviction policy

= Fast decoding
Greedy sampling of taking the highest probability token at each step leads to lack of diversity in the generated results

Nucleus sampling, otherwise known as top-$p$ sampling) considers multiple candidates during generation by setting a cumulative probability threshold $p$, allowing for sampling within a certain range
- More flexible than top-$k$ sampling as the candidate pool's size is based on the model's certainty in nucleus sampling

Nucleus sampling process more precisely from #link("https://en.wikipedia.org/wiki/Top-p_sampling")[Wikipedia]:
- The model calculates the probabilities for all possible next tokens.
- The tokens are sorted by their probability in descending order.
- The nucleus is formed by selecting tokens from the top of the list until their cumulative probability exceeds the predefined threshold, p.
- The probabilities of tokens within this nucleus are then rescaled so that they sum to 1. All tokens outside the nucleus are discarded (given a probability of 0).
- The final next token is randomly sampled from this new, smaller distribution.

== Speculative decoding
Particularly useful when generating low-entropy tokens e.g. in coding, which is very structured

Draft model quickly generates multiple candidate words, which are evaluated in parallel by the main/large model, calculating their probabilities/scores

Common draft model choices include a specific layer from the Transformer model, leveraging existing architecture for feature extraction, or training a smaller, separate model

== Skip layer
Dynamically and selectively skip certain layers during model inference
- Model evaluates the importance of each layer for the current task
- Optimization to learn to skip layers required during training through policy gradients (or other RL methods)

== CPU
ML-SpecQD uses multiple layers of draft models

== GPU
Multiple draft predictions made in parallel, and the longest prefix verified by the scoring model is used as the final output

Lookahead decoding uses Guess-and-Verify paradigm, generating draft tokens via $n$-gram, and verifying the draft tokens during forward simultaneously
- Additional tree-based attention mechanism to ensure correct relationship between draft tokens

Medusa adds extra decoding heads to generate multiple subsequent tokens in parallel

EAGLE uses a single transformer layer fro the LLM as the draft model, combining it with feature and token embeddings of the input

Ouroboros constructs a phrase candidate pool for draft token generation?

Sequoia uses dynamic programming and a hardware-aware tree optimizer to find the optimal tree structure based on runtime features and given hardware platform
- Novel sampling and verification method

Self-speculative decoding aims to go without the need of auxiliary models

Draft&Verify selectively skps certain intermediate layers

Kangaroo uses a fixed shallow sub-network of the LLM as the draft model, with the remaining layers serving as the target model, with an adapter module also trained to follow the sub-network  to enhance the representation ability of the draft model

LayerSkip proposes exit at early layers, verifying and correcting with remaining layers of the model
- At training time, dropout rate for later layers is higher

LLMA uses observation that there are abundant identical text spans between the decoding result by an LLM and the reference that is available in many real-world scenarios (e.g., retrieved documents)

Asynchronous Multi-device Speculative Decoding (AMUSD) decouples the draft and verify phases into a continuous, and asynchronous approach, with both models predicting independently on separate GPUs
- PipeSpec generalises this to $k$ models

Adaptix is a tri-gram matrix-based LLM representation to dynamically approximate LLM output distribution

SPIN dynamically picks between multiple small prediction models

Judge Decoding proposes a new judgement scheme that is more versatile so that poor alignment between draft and target does not lead to rejection of objectively correct continuations

Falcon proposes the Coupled Sequential Glancing Distillation technique to fortify inter-token dependencies within the same block during draft model training to increase speculation accuracy

AdaInfer statistically analyzes the activated layers across tasks, and uses simple algorithm to determine inference termination, but overhead makes this unsuitable for decoding
- RAEE builds on this with a retrieval database?

MOD decides whether to skip _current_ layer by pretraining the model to add a router in each layer like Mixture-of-Experts (MoE)

= Operator optimization
Operator fusion to reduce storage + transmission needs of intermediate data

Use linear approximations to nonlinear function approximation

Coarse-grained processing merges multiple fine-grained computation units to avoid frequent resource scheduling overheads and contention

Storage optimization to minimize latency by focussing on organizing data location and access patterns

== CPU
FlexInfer uses asynchronous prefetching, balanced memory locking and flexible tensor preservation to enhance memory efficiency and relieve bandwidth bottlenecks

== GPU
FlashAttention is an example of operator fusion
- FlashDecoding proposes additionally the parallel computation along the feature dimension
- FlashDecoding++ optimizes the synchronization overhead in softmax computation by pre-determining a unified maximum based on statistical analysis

FlashDecoding++ also introduces FlatGEMM that employs fine-grained tiling and double buffering techniques to improve parallelism and reduce the latency of memory access
- Also dynamically selects the most efficient operator based on the input with a heuristic

ByteTransformer and DeepSpeed uses operator fusion where main operator like GEMM is fused with lightweight operators e.g. residual adding, layer norm, and activation functions into a single kernel

SpInfer is a framework tailored for sparsified LLM inference on GPUs
- Integrates TensorCore-Aware Bitmap Encoding (TCA-BME) to minimize indexing overhead and integrates an optimized SpMM kernel with Shared Memory Bitmap Decoding (SMBD) and an asynchronous pipeline

FlashFormer exploits increased overlapping of memory movement for single-batch inference

Asynchronous KV cache prefetching

== FPGA
HAAN exploits correlation in normalization statistics among adjacent layers to bypass normalization computation by estimating statistics from preceding layers.

= Heterogeneous cooperation
Combine different computing platforms, to distribute tasks to the most suitable hardware

For heterogeneous cooperation, bandwidth often restricts computational performance
- High-speed on-chip and chip-to-chip interconnect e.g. Compute Express Link (CXL) and NVLink

== Homogeneous cooperation
Distributed computing like model parallelism aimed at addressing memory limitations

= Further discussion
Multimodality, inference-time compute (Chain-of-Thought (CoT)), and higher inference energy efficiency trends looking into the future

Inference time compute changes runtime breakdown of LLM inference
- Prefill stage much bigger part, with decode becoming a smaller part
- Prefill increases in part due to need to merge human input tokens with the template input tokens
- In the iterative process, intermediate output tokens are cumulatively fed back as inputs, increasing prefill length
- Introduction of Process Reward Model (PRM) accounts for 21.7% of the time
