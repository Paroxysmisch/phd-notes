#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://openreview.net/pdf?id=YFxfcQMLWX")[PADRe: A Unifying Polynomial Attention Drop-in Replacement for Efficient Vision Transformer]]

= Abstract
Several recent alternative attention mechanisms e.g. Hyene, Mamba, the original self-attention mechanism, etc. can be viewed as specific instances of PADRe

Key component is multiplicative nonlinearities, implemented with hardware-friendly operations such as Hadamard products
- Softmax avoided

= Introduction
Polynomial approximants in the input to replace standard self-attention
- Model capacity can be scaled-up efficiently by increasing the degree, while maintaining linear computation and memory

= Related work
== Efficient Vision Transformer Backbones
Recent designs use convolution early to downsample, then use transformers, but still quadratic, and becomes problematic with large inputs

Many alternative attention schemes, but all designed to process a single 2D image, and cannot be directly applied to more complex visual inputs e.g. 3D point clouds:
- ReLU-based attention
- Transposed attention
- Convolutional modulation
- Additive attention
- Shift-add attention
- Linear-angular attention

== Efficient attention drop-in replacements
Swin proposes windowed attention, but intricate reshaping and indexing pose practical challenges on resource-constrained hardware platforms

Linear cost attentions have been proposed, but require normalization factors at inference requiring summing over all tokens, which is inefficient on memory-constrained mobile platforms
- Some others require multiple heads to maintain accuracy

= PADRe framework approach
"Each element of the output tensor is a polynomial function of the elements of the input tensor, the coefficients of which are polynomial functions of the parameters (weights). Further, this dependency is such that the coefficients and the output itself may be computed efficiently (e.g., in linear time)."

Three main components:
+ Linear transformations
+ Non-linearities
+ Optional operations e.g. output resizing, normalization

== Linear transformations
$ "for" i = 1, ..., d, \

  Y_i = A_i X B_i
$, where $d$ is the degree of the framework, and $A_i$ and $B_i$ have structure e.g. convolutions imposed on them for efficient computation

Left- and right-multiplications create mixing among tokens (in the spatial dimension in the case of visual input), and in the embedding/channel dimension, respectively

== Non-linearities
Construct polynomial functions of the input

$
  Z_1 &= Y_1, \
  Z_(i+1) &= (C_i Z_i D_i) dot.o Y_(i+1), space "for" i in {1, ..., d-1}
$

The elements of $Z_i$ are homogeneous polynomials of degree $i$ in the input $X$

Dense, rather than homogeneous polynomials of the input computed, to increases expressivity:
$
  [P]_(m, n) = sum_(i=1)^d [W]_(m, n, i) [Z_i]_(m, n) + [L]_(m, n)
$
$L$ acts as a 0th-order term here

== Optional operations
$
  O = U P V
$
when the output of size different to the input is required

= Unifying framework
PADRe can be generalized to rational functions i.e. ratio of polynomials

Although standard self-attention cannot be represented by polynomials, it can be approximated by high-degree rational functions

= Experiments
Channel and token mixing operations implemented using pointwise convolutions or linear layers
- When input has inherent 2D structure e.g. image features, 11$times$11 2D convolutions to mix the tokens
- For sparse 3D voxels in a point cloud, they are treated as a sequence of tokens and 1D convolutions with kernel size 11 for mixing tokens

Single head used, as multiple heads did not provide empirical performance improvement

The design specified is a drop-in replacement for attention
- The degree-1 term is provided by the residual connection
- The degree-0 term is provided by the bias term in the convolutional layers

= Limitations and Future work
Need empirical results for cross-attention

Performance begins to saturate beyond degree 3---this could have to do with the numerical stability of monomials
- More stable polynomial basis, such as orthogonal polynomials could be used

Study PADRe's performance on sequential applications like NLP and time-series analysis
