#set text(font: "New Computer Modern")
#show heading: it => [#it #v(0.2em)]

#text(size: 2em)[#link("https://arxiv.org/pdf/2010.11929")[An Image is Worth 16$times$16 Words: Transformers for Image Recognition at Scale]]

= Introduction
Large scale training with lots of data trumps inductive bias---so we should focus on computationally efficient architectures that can learn inductive biases via lots of data

= Related work
Self-attention means that in the image domain, every pixel attends to every other pixel, which does not scale
- The local neighborhood self-attention approximation can replace convolutions
- Sparse transformers tried
- Applying attention in blocks of various sizes e.g. in extreme cases along individual axes
- All of these supposedly require complex engineering to be implemented efficiently in hardware

= Method
Model architecture is deliberately simple so that scalable NLP Transformer architectures can be used out of the box:
- Each image divided into patches
- Flatten each patch into a "token"
- Standard transformer encoder
- MLP classification head in the 0th token position
