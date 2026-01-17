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
== Logic synthesis
Accurate solutions required, so difficult to directly use ML algorithms to generate logic synthesis solutions, but we can use them to schedule existing traditional optimization strategies
- #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8942145")[LSOracle] uses DNN to dynamically decide which optimizer should be applied to different parts  of the circuit
  - This paper could potentially be improved through the use of GNNs, in particular hypergraph GNNs
  - The current DNN architecture works on a Karnuagh-map image (KMI) representation for circuit classification
  - Instead, a GNN could operate directly on the partitioned graphs

Like the above, there is #link("https://dl.acm.org/doi/epdf/10.1145/3195970.3196026")[additional similar work in this space]---these are called design flows where essentially transformations preserving the original logic are applied iteratively to simplify

Reinforcement learning has also been applied directly to the synthesis flow, only allowing valid transformations between DAGs as actions
- #link("https://ieeexplore.ieee.org/abstract/document/8351885?casa_token=VjXT--bj6CsAAAAA:ADc76tPptp7ETbY8hJ852B9Oa2NNB37YtZsO0i0EzLgIBHqjOfmx4bz6prFyW1mEH2Z4ienXZw")[This uses GCN for the policy function]

== Placement and routing prediction
#link("https://dl.acm.org/doi/epdf/10.1145/2228360.2228497")[PADE] enhances placement, especially regarding datapath placement
- The work currently uses quite old ML techniques, but we could enhance this by using GNNs directly on the netlist graph

Difficult to predict routing information in the placement stage
- Lots of work on congestion location prediction
- #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9045178")[This is just one paper of many], and uses GANs, but we can replace this with diffusion
- We can perhaps even incorporate the netlist information using GNNs---maybe #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7753259")[this] paper can help with incorporation of netlist information
- There looks to be #link("https://dl.acm.org/doi/pdf/10.1145/3400302.3415712")[some use of GNNs in this space already, but this paper is for cross-talk prediction only]

Simple regression tasks to predict final wirelength, timing performance, circuit area, power consumption, clock and other parameters

#link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8942063#page=2.83")[GAN-CTS] utilizes both GANs and RL for clock tree prediction and optimization, given placement information in the form of an image.
- Could we find a better way to represent placement information instead of an image?

Floorplanning is the preliminary step of the placement to roughly determine the geometric relationship among circuit modules and estimate the cost of the design---RL work here.

== Power Delivery Network (PDN) synthesis and IR (static and dynamic) drop prediction
Quite a lot of work involving CNNs like #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9045574")[this]
- I don't immediately see how we can bring a meaningful improvement here, apart from just improving the image model
- Possibly, we could use diffusion to generate the final output image/voltage map/etc.?

== Design challenges for 3D integration
Often stacking memory over logic

3D integration introduced die-to-die variation, which does not exist in 2D modeling, where the data or clock path may cross different dies in through-silicon via (TSV)-based 3D IC.
- Conventional modeling methods cannot accurately capture path delay
- Also expanded design space and additional overhead of design evaluation bring more challenges

Since VLSI circuits have a graph-like nature, #link("https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9218582")[GNNs have been used for tier partitioning in TP-GNN]

== Other predictions
Predict embedded memory timing failure during initial floorplan design

Work on aging effect prediction

== Summary
"In the physical design stage, recent studies mainly aim to improve the efficiency and accuracy by predicting the related information that traditionally needs further simulation. A popular practice is to formulate the EDA task as a computer vision (CV) task. In the future, we expect to see more studies that incorporate advanced techniques (e.g., neural architecture search, automatic feature generation, unsupervised learning) to achieve better routing and placement results."

= Lithography and mask synthesis
