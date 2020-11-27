## Generic

### qRNNs (ICLR 2017)

[Quasi-Recurrent Neural Networks](https://arxiv.org/abs/1611.01576), James Bradbury, Stephen Merity, Caiming Xiong, Richard Socher

- **Summary:** New approach for neural sequence modeling
- **Core idea:** Alternates convolutional layers with recurrent pooling functions
- **Key achievement:** Similar accuracy to the LSTM network but up to 16 times faster

### Asymmetric LSH (NIPS 2014)

[Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)](https://papers.nips.cc/paper/2014/file/310ce61c90f3a46e340ee8257bc70e93-Paper.pdf), Anshumali Shrivastava, Ping Li

- **Summary:** Sublinear time hashing algorithm for approximate Maximum Inner Product Search (MIPS)
- **Core idea:** Extend the LSH framework to allow asymmetric hashing schemes
- **Key achievement:** Efficient sublinear hashing scheme for MIPS

### ScaNN (ICML 2020)

[Accelerating Large-Scale Inference with Anisotropic Vector Quantization](https://arxiv.org/abs/1908.10396), Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar

- **Summary:** Quantization with anisotropic quantization loss functions leads to a new variant of vector quantization that more greatly penalizes the parallel component of a datapoint’s residual relative to its orthogonal component.
- **Core idea:** Score-aware quantization loss function
- **Key achievement:** A new quantization loss function for inner product search,  which replaces traditional reconstruction error
