#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link(
  "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11153923",
)[AI-Driven Integrated Circuit Design: A Survey of Techniques, Challenges, and Opportunities]]

= Introduction
Analog designs have sensitive trade-offs among multiple performance metrics

Validation done through computationally intensive SPICE and Electromagnetic (EM) simulations
- Particularly hard for RF and mmWave circuits, where layout-dependent parasitic effects dominate performance

CNNs and GNNs particularly successful in modeling high-frequency components that exhibit layout-dependent effects

= AI overview
Gradient-free metaheuristic algorithms:
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Multi-objective evolutionary algorithms (MOEAs) e.g. NSGA-II
- Used for sizing due to robustness to non-differentiable objectives---treat circuit simulators as closed box functions, relying solely on input-output evaluations without requiring analytical gradients or explicit internal models
- Suited to the high-dimensional, non-convex design spaces
- GA requires sampling an initial population, then using evolutionary operators primarily crossover and mutation
- In multi-objective optimization settings, GAs find a Pareto front of solutions
- Multi-Criteria Decision Making (MCDM) e.g. weighted sum, or Analytic Hierarchy Process(AHP) then used to select suitable design based on domain-specific priorities

Surrogate-based methods:
- Bayesian Optimization (BO)
- Strong performance in low-data regimes through model-based reasoning
- Beneficial when function evaluations are costly or limited

Reinforcement Learning:
- Handle sequential decisions
- Well suited for layout routing, or multi-step synthesis
- DRL pipelines can also incorporate supervised pretraining to accelerate convergence and stabilize early-stage learning
- Imitation learning can be used where the reward function is sparse or difficult to define
- Also Multi-Agent Reinforcement Learning (MARL), as well as Cooperative algorithms where agents can learn local policies while pursuing shared goals

Supervised Learning:
- Commonly used to regress performance metrics e.g. gain, power-added efficiency, and S-parameters directly from circuit design variables
- CNNs used when circuit design exhibits spatial structure e.g. in layout images or metal-dielectric patterns in EM simulations
- GNNs used on netlists

Unsupervised Learning:
- Design classification for grouping of designs with similar performance characteristics or identifiy distinct fault signatures
- Dimensionality reduction to make downstream optimization tasks more computationally tractable
  - E.g. the simplified representations often serve as an effective initialization mechanism for supervised or RL models

Generative models and representation learning
- Useful for tasks such as inverse design of passive components and data augmentation to reduce simulation costs
- LLMs trained on documentation and code capture complex design knowledge, guiding optimization algorithms
  - #link("https://dl.acm.org/doi/pdf/10.1145/3676536.3676816")[ADO-LLM] is an interesting paper which uses an LLM to help guide exploration in BO

= AI-Driven stages of circuit design and measurement
== Circuit topology and novel structure synthesis
AI-driven approaches have demonstrated uncovering topologies that surpass human-designed counterparts in terms of efficiency, compactness, and performance trade-offs (PPA)

Circuit optimization involves fine-tuning transistor-level parameters, such as channel dimensions, bias currents, and passive element values, to meet performance specifications such as gain, bandwidth, noise figure, and power efficiency.

#link(
  "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10937153",
)[Review of generative AI methologies, including diffusion models, for analog IC design]

#link("https://arxiv.org/pdf/2308.16406")[CktGNN] is a specific GNN architecture to work with circuits, and offers a benchmark regarding topology synthesis

== Circuit optimization
Circuit sizing is electing component values such as transistor widths and lengths, bias currents, resistor and capacitor values, and inductor geometries to satisfy specified performance targets (e.g., gain, bandwidth, noise, linearity, power consumption, and efficiency)
- Computationally expensive function evaluations often requiring SPICE-level circuit simulations of EM analysis

Evolutionary and Bayesian Optimization have a common limitation of the lack of adaptability to other circuit types or process technologies, and so the trained parameters to not generalize well across different design scenarios

RL has two primary advantages---its ability to learn generalizable strategies, and its flexibility to incorporate domain knowledge through reward shaping and state encoding
- GNNs for feature extraction from circuits viewed as graphs significantly help with generalizability
- Instead of full RL method, a potential idea could be training GNNs specifically for the purpose of extracting features from circuit-graphs?

Deep learning can be used to emulate the simulator, acting as a fast surrogate to predict performance metrics
- They can then be integrated with EAs, BO, or RL

Instead of optimizing the parameters of a predefined geometry (e.g., width and spacing of a inductor), inverse design aims to determine the necessary physical structure or component values required to achieve a specific set of desired performance characteristics
- This approach reframes the design challenge, moving directly from functional requirements to physical implementation
- Highly applicable to components where EM performance is strongly dictated by physical geometry e.g. antennas
- We want to learn a direct mapping from the performance domain to the design domain

Significant challenge is that a lot of the current work is *layout-unaware*, so due to issues such as layout parasitics, an optimal schematic design can be completely invalidated when it is time to perform layout
- Future work needs to tightly couple schematic sizing with physical layout generation
- E.g. transistor dimensions and placement needs to be optimized simultaneously
- #link("https://www.mdpi.com/2079-9292/12/2/465/pdf?version=1673917973")[GNN-based parasitic predictors] have been integrated directly into the optimization loop to provide layout-aware feedback at design time

== Circuit layout
Layout matters in analog or RFIC domain due to parasitic effects, signal coupling, and thermal considerations
- Additionally, we have considerations such as symmetry constraints, minimizing offsets, and managing signal path integrity to avoid crosstalk
- Also, manufacturability requirements

== Circuit testing and adaptation
