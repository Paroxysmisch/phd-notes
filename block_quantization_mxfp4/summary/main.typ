#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://arxiv.org/pdf/2511.04214")[Block Rotation is All You Need for MXFP4 Quantization]]

= Abstract
Methods like GPTQ delivery strong performance, but rotation-based approaches are incompatible with MXFP4

Mismatch between MXFP4's power-of-two block scaling and redestribution of outlier energy via global rotation

= Categorization of PTQ Methods
== Compensation-Based Quantization Methods
Adjust quantized weights to correct low-bit perturbations

GPTQ performs column-wise offline weight optimization using second-order information approximations from the Hessian
- Subsequent works are BoA, RSQ, QuantEase, VPTQ, APTQ

== Transformation-Based Methods
Equivalent transformations to redistribute or reshape data to reduce the impact of extreme values

SmoothQuant applies a smoothing transformation to redistribute large activations outliers to the corresponding weight scales, mitigating impact on low-bit quantization
- QServe, SmoothAttention build on this

QuIP introduces incoherent processing to decorrelate outlier contribution in weight and activation spaces
- QuIP\# uses randomized Hadamard transform to improve computational efficiency, and improves orthogonality and reduces inter-channel coherence

QuaRot and DuQuant use rotational transforms to spread outlier values across subspaces of smaller-magnitude activations

Transformation-based methods effective for modules exhibiting high activation variance or extreme outliers

== Optimization-Based Methods
Design equivalent transformations manually is hard, so we can parameterize them as learnable variables

OmniQuant introduces learnable weight clipping and equivalent transformations

SpinQuant has learnable rotation matrices

AffineQuant and FlatQuant extend by applying affine transformations to jointly adjust weights and activations, flattening distributions to mitigate outlier impact + simplifies optimization

KurTail leverages kurtosis-based rotation

= Evaluations
- INT4 benefits substantially from rotation. In the widely studied INT4 setting, applying rotation alone yields significant performance improvements. When combined with GPTQ or rotation optimization, the gains are further amplified, indicating that rotation is particularly effective for uniformly distributed integer formats.
- FP4 formats outperform INT4 without rotation. When not using rotation, BFP4 and MXFP4 achieve consistently higher performance than BINT4 and MXINT4, suggesting that FP4’s wider dynamic range and representational flexibility are better suited for 4-bit quantization.
- Random rotation degrades performance in group-wise formats. In contrast to INT4, groupwise quantization formats (BINT4, BFP4, MXINT4, MXFP4) suffer from performance degradation under random rotation. The effect is especially pronounced in MX-based formats, where performance can drop below that of simple RTN.
- Divergent behaviors under FP16 vs. PoT scaling. For FP16-scale formats, BINT4 outperforms BFP4 after random rotation. Conversely, for PoT-scale formats, MXFP4 underperforms compared to its FP16-scale counterpart, with rotation amplifying the discrepancy.
- PoT scaling in MX formats incurs additional loss. Comparing MXINT4/MXFP4 against their FP16-scale counterparts BINT4/BFP4, PoT scaling consistently introduces larger quantization errors, which become even more severe after rotation.
- Optimized rotation remains limited on MXFP4. While optimized rotation combined with GPTQ improves MXFP4 performance, the final results still lag behind those of INT4 under comparable configurations.

Rotation benefits INT4, but becomes harmful in MXFP4

= Why Rotation Transforms Hurt MXFP4?
= Limited recovery of large values in MXFP4 blocks
In both regular and outlier blocks, MXFP4's quantization error increases with the magnitude of the elements
- But much bigger problem for outlier blocks

PoT (Power-of-Two) scaling amplifies errors at large magnitudes, since it has coarse granularity at larger magnitudes

Main bottleneck of MXFP4 is in reconstructing large values in blocks, so improving MXFP4 performance depends on reducing these large values

= Rotation Induced Growth of Small Values
Rotation does note reduce overall energy as the L2 norm of the activations remains unchanged
- Thus, all previously small-valued channels become magnified as energy from the outlier is redistributed

Global rotation amplifies the scale of regular blocks, but since regular blocks vastly outnumber outlier blocks, the accumulated errors across them dominate

= Fix Rotation in MXFP4
Solution is to apply rotation transformations independently within each quantization block
- Global rotation mixes outliers across all channels
- Block-wise rotation partitions activations into fixed-sized groups that are aligned with MXFP4 blocks i.e. 32 channels
- Thus, outliers are only redistributed locally

Block-wise rotation is also more computationally efficient to implement
