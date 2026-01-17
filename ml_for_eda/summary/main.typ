#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://dl.acm.org/doi/pdf/10.1145/3451179")[ML for EDA]]

= Introduction
ML for EDA space covers almost all stages in the chip design flow: design space reduction + exploration, logic synthesis, placement, routing, testing, verification, and manufacturing.

Grouping of techniques in 4 categories (ordered by decreasing manual experience/human experties):
+ Decision making in traditional methods
  - Select among available tool chains/algorithms/hyper-parameters
  - Replace empirical choice/brute-force search
+ Performance prediction
  - Evaluate designs without synthesis---saves time
+ Black-box optimization
  - DSE
+ Automated design

Chip design flow:
+ System specification
+ Architectural design
+ Functional Design + Logic Design (RTL)
  - High-Level Synthesis (HLS) tools
+ Logic Synthesis (turn RTL into a circuit using logic gates)
  - Behavioural description turned into a gate-level description
  - Often takes a long time, so efficient DSE is important
  - Can use the result of the previous synthesis to guide the new one
  - Simulation can help verify design without having to synthesize
  - Cannot be solved optimally, so lots of heuristics used
+ Physical Design
  - Floorplanning
    - Better placement implies better chip area utilization, timing performance, and routability
    - Routing needs to satisfy timing performance and total wirelength requirements, without violating the design rules
    - Placement and routing are strongly coupled
  - Placement
  - Clock tree synthesis
  - Routing interconnections
  - PPA optimization
+ Physical Verification
  - ML methods for optimizing test set generation for verification
  - Want high test coverage, with high efficiency
  - Work on test set optimization and test complexity reduction
+ Fabrication
  - Mask synthesis is the key step, where mask optimization and lithography simulation is used (computational lithography?)
+ Packaging and Testing
+ Chip

= High Level Synthesis
== Result estimation
Acquiring accurate result estimation at HLS stage is difficult due to complex optimizations in the downstream physical synthesis stage
- Trade-off between efficiency and accuracy

ML methods to predict timing, resource usage, and operation delay from HLS design
  - Features (such as clock periods, resource utilization + availability, etc.) extracted from HLS reports used as inputs and target is to predict the implementation reports
  - Application characteristics, and target FPGA specifications also used as inputs
  - Also possible in an active learning setting where the predictions are used to inform new designs that are tested, to produce data to train the predictor, to refine predictions

Work done on performing better operation delay estimation
  - Simple additive delay estimation is inaccurate due to the post-implementation optimizations
  - #link("https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/abs/10.1145/3400302.3415657&hl=en&sa=T&oi=gsr-r&ct=res&cd=0&d=8752384747588768683&ei=_4xrab7sLNaOieoPnNOJuQQ&scisig=AHkA5jTQfFZ9mNScJHY_Es9kj598")[A customized GNN model is built to capture the association between operations from the dataflow graph, and train this model to infer the mapping choices about hardened blocks.]

Cross-platform performance estimation e.g. to estimate the speedup of an application for a target FPGA over an ARM processor
- Could be useful when porting an ML model to different accelerator hardware
- We can avoid FPGA synthesis and direct execution this way

= DSE in HLS
Pragmas in HLS are tunable synthesis options
- We often want to find the Pareto Frontier Curve, on which every point is not fully dominated by any other points under all the metrics

Improving conventional algorithms such as simulated annealing through initial point selection, generation of new samples, and hyper-parameter selection
- #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8587765")[STAGE] allows picking of initial points
- It is used for #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7556373")[DSE in 3D Network-on-Chip (NoC)], but I think we can take a radically different approach here, and form this as a GNN link prediction problem. Potential use case for hypergraphs too. Also potential for the use of graph diffusion. This is a constrained optimization problem where one of the key constraints is communication frequency.
- #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8119216")[Another NoC optimization paper]

= Logic Synthesis and Physical Design
