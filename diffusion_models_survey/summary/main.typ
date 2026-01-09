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

== Stochastic Differential Equations (Score SDEs)
What does it mean to solve an ODE?
- Consider ODE $(d x)/(d t) = f(x, t)$, $x(t_0) = x_0$.
- Solution is a function $x:[t_0, T] -> bb(R)^d$ such that:
  - $x(t)$ is differentiable,
  - it satisfies the equation pointwise: $d/(d t) x(t) = f(x(t), t) forall t$,
  - and satisfies the initial condition.

What does it mean to solve an SDE?
- Consider SDE $d X_t = f(X_t, t) d t + g(t) d W_t$, $X_t_0 = X_0$.
- Solution is *not* a function, but a stochastic process ${X_t}_(t in [t_0,T])$, with each realization being a random path.
  - The solution can be viewed as a probability law over paths, not a single curve.
- Often, the solution is expressed in integral form (we get by simply integrating both sides of the original SDE), in this case $X_t = X_t_0 + integral_(t_0)^t f(X_s, s) d s + integral_(t_0)^t g(s) d W_s$.

The "marginal" refers to the probability distribution of the random variable $bold(x) (t)$ at a single time $t$.

When we say that the ODE has the same marginals as the SDE, we mean $forall t: cal(L) (X_t^"ODE") = cal(L) (X_t^"SDE")$, where:
  - $X_t^"SDE"$ is the random variable at time $t$ from the SDE.
  - $X_t^"ODE"$ is the random variable at time $t$ obtained by:
    - sampling a *random initial condition*,
    - then evolving it *deterministically* via the ODE.
  - It does not mean that the ODE paths coincide with SDE paths, nor does it mean that for every SDE sample path there exists an ODE path that matches it.
  - An SDE is *stochastic* as randomness is added to it on "every infinitesimal step" by another random variable---e.g. $bold(w)$, which is the standard Wiener process a.k.a. Brownian motion.
  - With ODEs, randomness only appears once---that is, with the initial condition, with the evolution from that point being deterministic, but we can of course describe this evolution with a random variable encapsulating the random start.

For diffusion models, the core use case of the reverse SDE/ODE is generating a sample from the data distribution, starting from random noise.
- However, sampling with the reverse SDE injects noise at each step, and so the diffusion process cannot be reversed deterministically.
- Non-deterministic reversal is fine for generation, but problematic for likelihood evaluation, invertible mappings, and deterministic control.
- Reverse ODE solution is deterministic (apart from the initial seed, of course), and has (possibly) lower-variance generation.

At a high-level, numerical ODE solvers work by essentially hill-climbing on the gradient, starting from an initial point (given by the initial condition(s)).
- For example, given an ODE $(d x)/(d t) = v(x, t)$, $x(t_0) = x_0$, a numerical ODE solver constructs an approximate trajectory of points.
- Time is discretized: $t_0 < t_1 < ... < t_N$
- The solver advances: $x_(n+1) approx x_n + integral_(t_n)^(t_(n+1)) v(x(t), t) d t$.
- The integral is approximated.

Score SDE is the generalization of DDPMs and SGMs to the case of infinity time steps or noise levels, where the perturbation and denoising processes are solutions to SDEs.

Data is perturbed to noise with a diffusion process governed by the following forward SDE: $d bold(x) = bold(f) (bold(x), t) d t + g(t) d bold(w)$, where $bold(f) (bold(x), t)$ and $g(t)$ are diffusion and drift functions, with bold(w) being a standard Wiener process.
- The forward processes in DDPMs and SGMs are both discretizations of this SDE (the following hold as the total number of timesteps $T -> infinity$).
  - For DDPM, $d bold(x) = -1/2 beta(t) bold(x) d t + sqrt(beta(t)) d bold(w)$.
  - For SGM, $ d bold(x) = sqrt(d[sigma (t)^2]/(d t)) d bold(w)$. Since in SGM we generate the noise sample of the $t$th timestep directly from $bold(x)_0$, the diffusion process is not dependent on $bold(x)$.

For any diffusion form given by the forward SDE above, it can be reversed by solving the following reverse-time SDE: $d bold(x) = [bold(f) (bold(x), t) - g(t)^2 nabla_x log q_t (bold(x))] d t + g(t) d overline(bold(w))$, where $overline(bold(w))$ is a standard Wiener process when time flows backwards.
- Note that the solution trajectories of this reverse SDE shares the same marginal densities as those of the forward SDE.
- *Importantly*, there exists the probability flow ODE, whose trajectories have the same marginals as the reverse-time SDE: $d bold(x) = [bold(f) (bold(x), t) - 1/2 g(t)^2 nabla_x log q_t (bold(x))] d t$.
- Having the same marginals is important, as it allows sampling from the same data distribution.
- $q_t (bold(x))$ denotes the distribution of $bold(x)_t$ in the forward process.

We see that in both the reverse-time SDE and reverse-time ODE, we must know the score function at each time step $t$, $nabla_x log q_t (x)$.
- Solving the ODE via numerical techniques such as annealed Langevin dynamics, numerical SDE/ODE solvers, and predictor-corrector methods (MCMC + numerical ODE/SDE solvers) amounts to sampling a trajectory a.k.a. generating a sample.
- *Neural networks play the role of approximating this score function (like in SGMs).*
- To estimate the score function, we generalize the SGM score matching objective to continuous time: 
$ EE_(t ~ cal(U)⟦0, T⟧, x_0 ~ q(x_0), x_t ~ q(x_t | x_0)) [lambda(t) ||s_theta (x_t, t) - nabla_(x_t) log q_t(x_t | x_0)||^2] $

= Diffusion models with efficient sampling
Sampling involves a large number of evaluation steps, and can be sped-up by learning-free, and learning-based approaches (learning as an additional process after the diffusion model has been trained).

== Learning-free sampling
Focus on reducing the number of discrete time steps, while minimizing discretization errors.

=== SDE solvers
Since the DDPM generatino process can be viewed as a particular discretization of the reverse-time SDE, we can apply SDE solving techniques.
  - Noise-Conditional Score Networks (NCSNs) and Critically-Damped Langevin Diffusion (CLD) both solve the reverse-time SDE with inspirations from Langevin dynamics.
  - In particular NCSNs leverage annealed Langevin dynamics (ALD).
  - Sampling trajectories of ALD are not exact solutions to the reverse-time SDE, but have correct marginals, producing correct samples.
  - ALD further improved by Consistent Annealed Sampling (CAS), a score-based Markov Chain Monte Carlo approach, with better scaling of time steps and added noise.

One-step discretization of the forward SDE has the general form: $bold(x)_(i+1) = bold(x)_i + bold(f)_i (bold(x)_i) + bold(g)_i bold(z)_i$, $i = 0, 1, ..., N-1$, where $bold(z)_i ~ cal(N) (bold(0), bold(I))$, $bold(f)_i$ and $bold(g)_i$ are determined by drift/diffusion coefficients of the SDE and the discretization scheme.

Reverse-time SDE diffusion discretization: $bold(x)_i = bold(x)_(i+1) - bold(f)_(i+1) (bold(x)_(i+1)) + bold(g)_(i+1)bold(g)_(i+1)^t bold(s)_theta (bold(x)_(i+1), bold(t)_(i+1)) + bold(g)_(i+1) bold(z)_i$, $i = 0, 1, ..., N-1$, which is similar to the discretization of the forward SDE.

Another proposal is adaptive step sizes. The output of higher- and lower-order SDE solvers is compared. If both their outputs are similar, then the higher order result is used for the next denoising step (since it's using higher-order terms, and so is more accurate), and the step-size is increased.

Predictor-corrector methods use the SDE solvers as the predictor, and iterative MCMC as the corrector. Numerical SDE solver produces course sample that is then corrected with score-based MCMC, which corrects the sample's marginal distribution.

=== ODE Solvers
Converge much faster than SDE solvers, but inferior sample quality.

Denoising Diffusion Implicit Models (DDIM) originally motivated to extend DDPM to a non-Markovian case: $q(bold(x)_1, ..., bold(x)_T|bold(x)_0) = product_(t=1)^T q(bold(x)_t|bold(x)_(t-1), bold(x)_0)$ (Further details in DDIM paper).
  - This formulation captures DDPM and DDIM as special cases.
  - When $sigma_t^2 = 0$, the Markov chain learnt by DDIM to reverse this non-Markov diffusion process is fully deterministic.
  - From DDIM, we also have generalized Denoising Diffusion Implicit Models (gDDIM), allowing for more general diffusion processes like Critically-Damped Langevin Diffusion (CLD) and PNDM.

We can also use 2nd order methods, rather than Euler's method for sampling, which although requires an additional evaluation per timestep, leads to smaller discretization error, and thus samples of comparable/better quality, with fewer sampling steps.

Diffusion Exponential Integrator Sampler and DPM-solver are even higher order integrators.

== Learning-Based Sampling
This involves partial steps, or training a sampler for the reverse process.

Faster sampling speeds at the expense of slight degradation in sample quality.

Learning-based sampling typically involves selecting steps by optimizing certain learning objectives instead of the hand-crafted steps used in learning-free approaches.

=== Optimized Discretization
Select the best $K$ time steps to maximize the training objective for DDPMs. Key is the observation that the DDPM objective can be broken down into a sum of individual terms, making it well-suited for dynamic programming.
  - *However the variational lower bound used for DDPM training does not correlate well with sample quality.*
  - Differentiable Diffusion Sampler Search addresses this issue by directly optimizing a common metric for sample quality called the Kernel Inception Distance (KID), and is made feasible with reparameterization and gradient rematerialization.

=== Truncated Diffusion
Start reverse denoising process with a non-Gaussian distribution. Samples from this distribution can be obtained efficiently from other pre-trained generative models, such as Variational Autoencoders (VAEs) or Generative Adversarial Networks (GANs).

=== Knowledge Distillation
Progressive Distillation involves distilling the full sampling process into a faster sampler requiring only half as many steps.

