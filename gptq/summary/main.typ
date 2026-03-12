#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]
#set math.equation(numbering: "(1)")

#text(size: 2em)[#link(
  "https://arxiv.org/pdf/2210.17323",
)[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers]]

= Abstract
One-shot (post-training) weight quantization method

Based on approximate second-order information

Reduce bitwidth to 3--4 bits per weight

= Introduction
Also bespoke GPU kernels to leverage compression for faster memory loading

Method does not provide speedup for actual multiplications due to lack of hardware support for mixed-precision operands (e.g. FP16$times$INT4)

No activation quantization, as not a significant bottleneck in target scenarios

== Large-model quantization
LLM.int8() observes activation outliers in a few feature dimensions break quantization of larger models---fixed by keeping those dimensions in higher precision

= Background
Method performs layer-wise quantization so that:
$
  "argmin"_hat(W) ||W X = hat(W) X||_2^2
$ <eq:layer-wise-quantization>
Quantized weights $hat(W)$ minimize difference in activations per-layer

== Optimal Brain Quantization (OBQ)
+ @eq:layer-wise-quantization can be written as the sum of squared errors over each row of $W$
+ OBQ handles each row $w$ independently, quantizing one weight at a time while always updating all not-yet-quantized weights---compensates for error incurred by quantizing a single weight
+ Weights quantized iteratively using equations for the greedy-optimal weight to quantize next $w_q$, and the corresponding optimal update of all weights in $F$ (set of remaining full-precision weights) denoted by $delta_F$
  + Iteration stopped when all weights are quantized
+ Vectorized implementation handles multiple rows of $W$ in parallel
+ For matrix $W$ of size $d_"row" times d_"col"$, the complexity is cubic $cal(O)(d_"row" dot d_"col"^3)$

= GPTQ Algorithm
+ Relaxed ordering of weights to be quantized---any fixed ordering may perform well, especially on large models
+ Batching used to combat low arithmetic intensity of the algorithm on GPUs and make better use of the limited memory bandwidth
+ Cholesky reformulation (and dampening in smaller part) to deal with numerical inaccuracies

= Experimental validation
Each Transformer block, consisting of 7 layers processed at a time
- Current block inputs are sent through the fully quantized block again to produce the new inputs for the quantization of the next block
- So, the quantization process operates not on the layer inputs in the full precision model, but the actual layer input in the already partially quantized model

Perplexity-based tasks particularly sensitive to model quantization

Special kernel that performs matrix vector by dynamically dequantizing weights when needed
- Does not require any activation quantization

== Summary and Limitations
Method obtains speedups from reduced memory movement, and not computational reductions

Study focuses on generative tasks and does not consider activation quantization
