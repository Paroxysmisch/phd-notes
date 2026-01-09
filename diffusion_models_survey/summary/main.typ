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

== Score-Based Generative Models (SGMs)
For a probability density function $p(x)$, its score function (a.k.a. Stein score) is the gradient of the log probability density $nabla_x log p(x)$.
- Unlike Fisher score ($nabla_theta log p_theta (x)$), Stein score is a function of the data $x$.
- Stein score can be viewed as a vector field pointing along direction in data-space where the probability function has the largest growth rate.

With SGMs, we get the neural network to approximate this score function, conditioned on a particular noise level.
- Generating samples for inference involves chaining the score function neural approximation at decreasing noise levels.
- Sampling can be performed with various score-based sampling approaches, including Langevin Monte Carlo, Stochastic Differential Equations (SDEs), Ordinary Differential Equations (ODEs), and their various combinations.

To train the neural network, we take a real data point and perturb it with a sequence of intensifying Gaussian noise.
- Since we are noising the data (from the initial data point $bold(x)_0$, so $q(bold(x)_t|bold(x)_0) = cal(N) (bold(x)_t;bold(x)_0, sigma_t^2 I)$, we know the direction where the probability function has highest growth rate, and so we can perform supervised training with the neural network.
- Learning from perturbed data points can be done using score matching, denoising score matching, and sliced score matching.
- For example, with denoising score matching:
$ EE_(t ~ cal(U)⟦1, T⟧, x_0 ~ q(x_0), x_t ~ q(x_t | x_0)) [lambda(t) sigma_t^2 ||nabla_(x_t) log q(x_t) - s_theta (x_t, t)||^2] $

$ = EE_(t ~ cal(U)⟦1, T⟧, x_0 ~ q(x_0), x_t ~ q(x_t | x_0)) [lambda(t) sigma_t^2 ||nabla_(x_t) log q(x_t | x_0) - s_theta (x_t, t)||^2] + "const" $

$ = EE_(t ~ cal(U)⟦1, T⟧, x_0 ~ q(x_0), x_t ~ q(x_t | x_0)) [lambda(t) ||- (x_t - x_0) / sigma_t - sigma_t s_theta (x_t, t)||^2] + "const" $

$ = EE_(t ~ cal(U)⟦1, T⟧, x_0 ~ q(x_0), epsilon ~ cal(N)(0, I)) [lambda(t) ||epsilon + sigma_t s_theta (x_t, t)||^2] + "const," $
- If we set $bold(epsilon)_theta (bold(x), t) = - sigma bold(s)_theta (bold(x), t)$, we see that the training objectives of DDPMs and SGMs are equivalent ($bold(s)_theta (bold(x)_t, t)^2$ is just the neural network approximating the score function, while being conditioned on the timestep).

The key property about SGMs is that training and sampling are completely decoupled.

SGMs, like DDPMs, generate samples iteratively from $bold(s)_theta (bold(x), T), bold(s)_theta (bold(x), T-1), ..., bold(s)_theta (bold(x), 0)$.
- One method for sampling is annealed Langevin dynamics (ALD). Note that there are $N$ iterations *per timestep*, with $s_t > 0$ step size.
- ALD initialized to $bold(x)_T^((N)) ~ cal(N) (bold(0), bold(I))$, then applying Langevin Monte Carlo for $t = T, T-1, ..., 1$ for $0 <= t < T$.
- In Langevin Monte Carlo, we start with $bold(x)_t^((0)) = bold(x)_(t+1)^((N))$, with the following update rule for $i = 0, 1, ..., N-1$:
  - $bold(epsilon)^((i)) <- N (bold(0), bold(I))$
  - $bold(x)_t^((i+1)) <- bold(x)_t^((i)) + 1/2 s_t bold(s)_theta (bold(x)_t^((i)), t) + sqrt(s_t) bold(epsilon)^((i))$
- Langevin Monte Carlo guarantees that as $s_t -> 0$ and $N -> inf$, $bold(x)_0^((N))$ becomes a valid sample from the data distribution $q(bold(x)_0)$.
- We can view this sampling as hill climbing the probability density using its estimated gradient.






