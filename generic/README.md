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

### RigL (ICML 2020)

[Rigging the Lottery: Making All Tickets Winners](https://proceedings.icml.cc/static/paper_files/icml/2020/287-Paper.pdf), Utku Evci, Trevor Gale, Jacob Menick, Pablo Samuel Castro, Erich Elsen

- **Summary:** An algorithm for training sparse neural networks that uses a fixed parameter count and computational cost throughout training, without sacrificing accuracy relative to existing dense-to-sparse training methods

- **Core idea:** 1) Remove a fraction of the connections with the smallest weight magnitudes. 2) Activate new connections using instantaneous gradient information. 3) Keep training with the updated network until the next scheduled update. 4) Activate connections with large gradients, since these connections are expected to decrease the loss most quickly.

- **Key achievement:** Achieving state-of-the-art performance with much fewer parameters

### DrRepair

[Graph-based, Self-Supervised Program Repair from Diagnostic Feedback](https://arxiv.org/abs/2005.10636)

- **Summary:** Automatic program repair based on a program-feedback graph and a graph neural network for reasoning

