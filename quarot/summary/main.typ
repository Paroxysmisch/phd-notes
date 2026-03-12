#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link(
  "https://arxiv.org/pdf/2404.00456",
)[QuaRot: outlier-Free 4-Bit Inference in Rotated LLMs]]

= Abstract
Quantize all weight, activations, and KV cache in 4 bits

Removes outliers from the hidden state without changing the output

= Introduction
Quantizing activations is hard as they have large outlier elements (in channels)

Previous work uses calibration set to characterize outlier features and keeps them in higher precision for inference

This work uses randomized Hadamard transformations
- Computational invariance idea used so that Hadamard transformations are fused into weight matrices
- We get an equivalent network without outlier features that can operate on activations in this Hadamarded space

= Related work
LLM.int8() identifies outlier features during inference, keeping them in 16-bits, but poor performance

SmoothQuant normalizes features with scaling factors from a calibration set, but extra hyper-parameters

Atom develops complex kernel for mixed-precision MatMul in the presence of outliers

Computational invariance theorem from #link("https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2401.15024&hl=en&sa=T&oi=gsr-r&ct=res&cd=0&d=6903691151937521935&ei=hL-yaY7ND8-NieoP9e6boQs&scisig=AFtJQix0GBi4pK9kB2ffmicZsPTP")[SliceGPT]
- Transformations are fused into the weights

= Background
Rotational matrices are orthogonal matrices

Orthogonal matrices with determinant 1 are rotational matrices

== Incoherence processing
Multiplying weight matrices by orthogonal matrices improves incoherence

This technique can also be applied to activation quantization

= Method
GPTQ used to quantize weights

Activations quantized on-the-fly using simple round-to-nearest scheme
