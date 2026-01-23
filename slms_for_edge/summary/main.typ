#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]
#let highlight(content) = block(
  fill: yellow.lighten(70%),
  inset: 1.5em,
  radius: 1.5em,
  stroke: yellow.lighten(20%),
  breakable: true,
)[#content]
#show link: set text(fill: blue)

#text(size: 2em)[#link(
  "https://aclanthology.org/2025.acl-long.718.pdf",
)[SLMs for Edge]]

= Abstract
SLMs have limited in-context learning capabilities

Significant potential for efficiency optimization
- Dynamic task-specific routing
- Model-hardware co-design
- Vocabulary/KV cache compression

= Introduction
100M--5B parameters in decoder-only transformer architecture

SLMs are typically trained on many more tokens that what Chinchilla recommends i.e. "over-trained"

Dataset quality is more important than dataset size

Model architecture has a meaningful impact on inference speed

Quantization gains diminish in long context

= SLM Overview and Benchmarking
== SLM Capabilities
Gap between open-source and closes-source SLMs is narrowing, driven by high-quality datasets like DCLM and FineWeb-Edu

Small models can excel in specific tasks, out-performing models with even double the parameter count
- Lots of potential for task-specific model routing--great for efficiency and accuracy

In-context learning capabilities (ICL)
- Benchmarks with lower complexity don't benefit as much from multiple shots during prompting, as we would expect
- Most models improve with multiple shots, except for LaMini, which could be due to over-fitting, where additional context introduces noise
- ICL improves with model size

== Training datasets
Many pre-training datasets contain coding data, driven by the belief that coding data enhances reasoning ability

Chinchilla law suggests optimal parameter-to-token ratio of 1:20, but recent models use significantly more tokens
- This strategy aims to enhance SLM performance for resource-constrained deployment by increasing training-time FLOPs, but over-training can lead to performance saturation

Model-based filtering (where another model is used to automatically filter, score, or select training data instead of relying only on heuristics of human annotation) is a key trend in dataset research, and aims to improve data quality which is very important
- Remove noisy, misaligned, or harmful data

= SLM Runtime Cost
== Inference latency
Pre-fill stage dominates on-deice LLM inference due to long-context processing for personalization on edge

Allocating more parameters to attention and FFN increases computational cost

Linear trend of decode latency per token with model parameter count, and is primarily memory-bound, unlike the compute-bound prefill stage, as:
- We get excellent parallelism during prefill.
- For decode, we just calculate *Q* for the new token, but need to load all past *K* and *V* vectors. Then attention is computed against the full history, and we write one new KV entry. The dominant cost is reading $cal(O)(L dot d)$ KV cache from memory, with very few FLOPs per byte loaded
- Larger models have a larger hidden dimension $d$, increasing the size of the KV cache per token, which means more memory traffic generated per token

Prefill benefits from:
- Faster compute (FLOPs)
- Better kernels
- Quantized matmuls

Decode benefits from:
- Smaller KV caches
- KV quantization
- FlashAttention / paged attention
- Shorter contexts

This is why many edge optimizations target decode, not prefill.

Architectural differences impact compute-bound stages more significantly
- Wider, shallower models benefit from higher parallelism

== Memory footprint
Memory scales linearly with model size, with vocab size having a disproportionately large impact
- Using GQA reduced KV cache requirements

Especially for smaller models, model architecture has a greater impact than model size for inference latency---likely hardware-dependent correlation
- Since pre-fill is compute-bound, impact of model architecture on inference speed more significant at prefill stage

SLM architecture should align with hardware design, optimizing vocabulary size, FFN width, and layer depth for efficiency
- Edge devices should leverage hardware heterogeneity, using NPUs for prefill and CPUs for decode

== Impact of quantization
Quantization reduces memory access overhead

We can also lower memory usage, leading to higher batch size

We load/store in integer, and activations/initial input is in floating point
- We actually multiple a floating point activation by integer weights, then rescale to get the final activations
- This ensures stability as activations have dynamic ranges that change per token, and quantizing activations aggressively hurts accuracy
- Also, decode is memory bound, so shrinking weight size is more important than shrinking compute
- LayerNorm computations, softmax, residual connections, and attention scores usually not quantized
- The KV cache is an exception as it is quantized despite it being activations

We use integer quantization to go below 8-bit quantization effectively, as there is no FP4 in mainstream hardware
- Also integer quantization allows us to have higher effective precision as since precision in a block of weights is uniform, by using the same scale value for the block, we can use all of the integer bits to store what is essentially the mantissa
- We essentially store one scale per block, amortizing the float exponent overhead
- Loading INT4 weights, then expanding in registers is still faster than loading FP16 weights directly

With longer prompts, the impact of quantization diminishes as the cost of moving the weights into memory is amortized across more tokens

In the decode stage however, quantization provides more consistent improvements as weights are accessed per-token

Irregular bit-widths cause performance regressions from just using FP16

== Impact of hardware
While the GPU is $40 times$ faster than the CPU during pre-fill, it is only $1.84 times$ faster during decode

== Latency breakdown
The latency split between attention, FFN, and LM Head (the three main contributors) is roughly the same between prefill and decode

Vocabulary size significantly influences memory consumption beyond model size

At long contexts, compute buffer and KV cache dominate memory usage
