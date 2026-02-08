#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link(
  "https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf",
)[High-Resolution Image Synthesis with Latent Diffusion Models]]

= Abstract
Cross-attention layers turn diffusion models (DMs) into generators for general conditioning inputs such as text or bounding boxes

= Introduction
GANs mostly confined to data with comparably limited variability as adversarial training does not easily scale to modeling complex, multi-modal distributions
- Generator's goal is fool discriminator, so it may focus entirely on producing one animal, for example, instead of all animals
- If discriminator becomes too good too quickly, gradient becomes flat, so generator receives no useful feedback on how to improve
  - With a multi-modal distribution, generator is further from truth---equivalent to discriminator being relatively better
- Unlike Variational Autoencoders (VAEs) or diffusion models, GANs never see the whole probability distribution---only learn through the critiques of the discriminator
- Likelihood-based models don't exhibit mode-collapse and training instabilities of GANs

Likelihood models can exploit parameter sharing to model complex distributions without billions of parameters as in auto-regressive (AR) models
- We can use networks like CNNs that have parameter sharing as working on the entire image at once
- Hierarchical Understanding---a Diffusion model uses downsampling to see the big picture and upsampling to see the details. It doesn't need a separate parameter for every pixel-to-pixel relationship; it just needs to learn how to de-noise a patch of pixels, a task that is mathematically similar whether it's in the corner or the center

Likelihood-based models, like DMs, spend excessive capacity modelling imperceptible details of the data
- Reweighted variational objective tries to address this by undersampling initial denoising steps

Two distinct phases of training
+ Autoencoder providing lower-dimensional perceptually equivalent representational space to the data space
+ DM training

Separation ensures that delicate weighting of reconstruction and generative abilities is not required, and we can train various diffusion models for a given, universal autoencoding stage

= Method
Although DMs allow ignoring perceptually irrelevant details by undersampling the corresponding loss terms, they still require costly function evaluations in pixel space

== Perceptual image compression
Autoencoder trained by perceptual loss + patch-based adversarial objective

Downsamples by a factor of $f = H/h = W/w$, and a variety of downsampling factors investigated $f = 2^m$

Latent space regularization through 2 competing methods:
- KL-reg---slight KL-penalty towards standard normal
- VQ-reg that is a vector quantization layer within the decoder that snaps the continuous encoder output to the closest fixed point in a learnable codebook
  - The snapping is non-differentiable, so a Straight-Through Estimator (STE) trick is used where we use the gradient of the loss with respect to the quantized/snapped code as the loss for the encoder output
  - Forward: $z_q = "quantize" (z_e)$
  - Backward: $(partial L) / (partial z_e) = (partial L) / (partial z_q)$
  - Another alignment term in loss to move codebook closer to encoder output, and also move the encoder outputs closer to the codebook via a commitment loss
  - Prevents model from overfitting to noise

Latent space is still 2D

== Conditioning mechanisms
We are implementing a conditional denoising autoencoder $epsilon.alt_theta (z_t, t, y)$ for $y$ such as text

DMs turned into more flexible conditional image generators (that just conditioning on things like class labels) by augmenting the UNet backbone with cross-attention

Domain-specific encoder $tau_theta$ projects $y$ to an intermediate representation $tau_theta (y) in RR^(M times d_tau)$
- Mapped to intermediate layers of the UNet via cross-attention:
$
  Q = W_Q^((i)) dot phi_i(z_t), #h(2em) K = W_K^((i)) dot tau_theta (y), #h(2em) V = W_V^((i)) dot tau_theta (y)
$
- Each latent "pixel" forms a query

= Experiments
VQ-regularized latent spaces achieve better sample quality, although the autoencoder's reconstruction quality is reduced, which seems paradoxical, but the latent space is simpler for the diffusion model to navigate

Autoencoder part is the perceptual compression, and diffusion part is the semantic compression (composition of the data)

= Conditional latent diffusion
== Transformer encoders for LDMs
Classifier-Free Guidance (CFG) to allow the model to more closely follow say a text prompt, and greatly boosts sample quality
- Without it, model might produce a generic image only vaguely resembling the prompt as it tried to balance the instructions with what it knows about images in general
- With CFG, we can prioritize the signal of the prompt
- $hat(epsilon.alt)_theta (x_t, c) = epsilon.alt_theta (x_t, nothing) + omega dot (epsilon.alt_theta (x_t, c) - epsilon.alt_theta (x_t, nothing))$
- $omega$ controls how much we amplify the difference between the conditioned and unconditioned prediction
- The Old Way (Classifier Guidance): You had to train a separate, second model (a classifier) that could look at a noisy image and tell you if it contained a dog or a cat. During diffusion, use that second model's gradients to nudge the image. But, this means you had to train a separate model for every concept, and it often resulted in adversarial hacks where the image looked like noise to humans but dog to the classifier.
- The New Way (Classifier-Free): You use the same model for everything. During training, you randomly hide the text prompt (nulling it out) about 10--20% of the time. This teaches the model how to function both as a generic generator and a conditioned one.
