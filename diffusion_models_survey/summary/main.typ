#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://dl.acm.org/doi/pdf/10.1145/3626235")[Diffusion Models: A Comprehensive Survey of Methods and Applications]]

= Abstract
Discusses efficient sampling, improved likelihood estimation, and handling data with special structure.

Combining diffusion with other generative models.

Applications, including NLP.

Taxonomy found at https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy.

= Introduction
Interesting additional areas where diffusion is applied are:
- temporal data modelling
- multi-modal
- robust machine learning (creating models that maintain stable, reliable performance even when facing unexpected variations, noisy data, distribution shifts, or deliberate adversarial attacks, ensuring they work consistently in real-world, unpredictable conditions, not just on clean training data)

= Foundations of diffusion models
== Denoising Diffusion Probabilistic Models (DDPMs)
Forward (noising) and reverse (denoising) Markov chains.

Forward hand-designed to transform ANY data distribution into a simpler, tractable prior distribution.
- Transition kernel denoted with $q(bold(x)_t|bold(x)_(t-1))$.
- Gaussian perturbation commonly used, and has the nice property that it allows for easy marginalization of the joint forward distribution (of all the intermediate states) to allows us to jump from $bold(x)_0$ to $bold(x)_t$ for $forall t$ in analytical form.
- $beta_t$ can be seen as how much information to remove in each timestep.
- Thus $overline(a)_t$ can be seen as the cumulative information retained after $t$ denoising timesteps.

Key to success of the learnable transition kernel $p_theta (bold(x)_(t-1)|bold(x)_t)$ sampling process is training the reverse Markov chain to match the actual time reversal of the forward Markov chain.
- $p_theta (bold(x)_0, bold(x)_1, ..., bold(x)_T) approx q(bold(x)_0, bold(x)_1, ..., bold(x)_T)$ 
- This is achieved by minimizing the KL-divergence between these two.
- The evidence is the marginal likelihood of the observed data $bold(x)_0$ under our estimated distribution $log p_theta (bold(x)_0)$, but this is intractable to compute as it requires integrating over all latent trajectories $x_(1:T)$. We want to maximize this evidence.
- Some notation: $p_theta (bold(x)_(0:T)) eq.triple p_theta (bold(x)_0, bold(x)_1, ..., bold(x)_T)$, and in integrals, $d x_(1:T) eq.triple d x_1 d x_2 ... d x_T$ and is shorthand for integrating over all latent variables.
- $ log p_theta(x_0)
  &=
  log integral p_theta(x_(0:T)) dif x_(1:T)
  =
  log integral
    q(x_(1:T)|x_0)
    frac(
      p_theta (x_(0:T)),
      q(x_(1:T)|x_0)
    )
  dif x_(1:T) \
  &=
  log (E_(q(x_(1:T)|x_0))[
    frac(
      p_theta(x_(0:T)),
      q(x_(1:T)|x_0)
    )
  ])
  >=
  E_(q(x_(1:T) | x_0))[ log
    frac(
      p_theta(x_(0:T)),
      q(x_(1:T)|x_0)
    )
  ]
$
- The above ELBO is essentially re-expressing the calculation of the evidence using important sampling from a distribution we can actually sample from i.e. the forward distribution. The final step is applying Jensen's inequality.
- Why do we need Jensen's inequality? If we directly use the evidence importance sampling estimator $(log (E_(q(x_(1:T)|x_0))[
    frac(
      p_theta(x_(0:T)),
      q(x_(1:T)|x_0)
    )
  ]))$, it's variance explodes as we are taking the samples (to approximate the expectation) over a chain of events. Thus, we push the $log$ inwards, and this has the effect of giving us a lower bound on the true evidence, and also makes it an unbiased estimator.
- $ p_theta(x_(0:T))
  =
  p(x_T)
  product_(t=1)^T p_theta (x_(t-1)|x_t)
$
$ q(x_(1:T)|x_0)
  =
  product_(t=1)^T q(x_t|x_(t-1))
$
- Substitute into ELBO above and simplify: $ "ELBO"
  &=
  E_q [
    log p(x_T)
    +
    sum_(t=1)^T log p_theta (x_(t-1)|x_t)
    -
    sum_(t=1)^T log q(x_t|x_(t-1))
  ] \
  &=
  - E_q [
    - log p(x_T)
    -
    sum_(t=1)^T
      log frac(
        p_theta (x_(t-1) | x_t),
        q(x_t | x_(t-1))
      )
  ]
$
- The conservatism (under-estimating the evidence) also disappears as the model improves as $log p_theta (bold(x)_0) - "ELBO" = "KL"(q(bold(x)_(1:T)|bold(x)_0)||p_theta (bold(x)_(1:T)|bold(x)_0))$.

Various terms in the Variational/Evidence Lower Bound can also be reweighted for better sample quality.

$bold(epsilon)_theta (bold(x)_t, t)$ represents the model predicting the noise vector $bold(epsilon)$ given $bold(x)_t$ and $t$.

Commonly used training objective, with a positive weighting function $lambda (t)$ is: $bb(E)_(t ~ cal(U)[[1, T]], bold(x)_0 ~ q(bold(x)_0), bold(epsilon) ~ cal(N)(bold(0), bold(I))) [lambda (t)||bold(epsilon) - bold(epsilon)_theta (bold(x)_t, t)||^2]$.





