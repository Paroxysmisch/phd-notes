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

= Background

