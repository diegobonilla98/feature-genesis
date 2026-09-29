# Primary References

## Dataset

Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. *The Caltech-UCSD Birds-200-2011 Dataset*. California Institute of Technology Technical Report CNS-TR-2011-001, 2011.

- Official dataset page: https://www.vision.caltech.edu/datasets/cub_200_2011/
- Official data record: https://data.caltech.edu/records/65de6-vp158

## Sparse autoencoders

Bart Bussmann, Patrick Leask, and Neel Nanda. *BatchTopK Sparse Autoencoders*. arXiv:2412.06410, 2024.

- https://arxiv.org/abs/2412.06410

Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders*. arXiv:2407.14435, 2024.

- https://arxiv.org/abs/2407.14435

Bart Bussmann, Noa Nabeshima, Adam Karvonen, and Neel Nanda. *Learning Multi-Level Features with Matryoshka Sparse Autoencoders*. arXiv:2503.17547, 2025.

- https://arxiv.org/abs/2503.17547

Valérie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, and Demba Ba. *From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit*. arXiv:2506.03093, 2025.

- https://arxiv.org/abs/2506.03093

Silen Naihin and Lev Stambler. *Size Doesn't Matter: Cosine-Scored Sparse Autoencoders*. arXiv:2606.15054, 2026.

- https://arxiv.org/abs/2606.15054

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. arXiv:2309.08600, 2023.

- https://arxiv.org/abs/2309.08600

## SAE evaluation and limitations

Jonas Klotz, Cassio F. Dantas, Pallavi Jain, Diego Marcos, and Begüm Demir. *Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations*. arXiv:2606.24716, 2026.

- https://arxiv.org/abs/2606.24716

David Chanin et al. *SynthSAEBench: Evaluating Sparse Autoencoders on Realistic Synthetic Data*. arXiv:2602.14687, 2026.

- https://arxiv.org/abs/2602.14687

Alexander Paulo, Alex Mallen, C. Daniel Freeman, and Jacob Steinhardt. *Sparse Autoencoders Do Not Find Canonical Units of Analysis*. arXiv:2502.04878, 2025.

- https://arxiv.org/abs/2502.04878

David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, and Joseph Bloom. *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders*. arXiv:2409.14507, 2024.

- https://arxiv.org/abs/2409.14507

## Training-data attribution

Chirag Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. *Estimating Training Data Influence by Tracing Gradient Descent*. NeurIPS, 2020.

- https://arxiv.org/abs/2002.08484

Sung Min Park et al. *TRAK: Attributing Model Behavior at Scale*. ICML, 2023.

- https://arxiv.org/abs/2303.14186

Pang Wei Koh and Percy Liang. *Understanding Black-box Predictions via Influence Functions*. ICML, 2017.

- https://arxiv.org/abs/1703.04730

## Reproducibility

PyTorch. *Reproducibility*. Official documentation.

- https://docs.pytorch.org/docs/stable/notes/randomness.html

PyTorch. *torch.use_deterministic_algorithms*. Official documentation.

- https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html

## Methodological note

The repository implements the core mathematical ideas directly, but it is not presented as an official reproduction of any one paper's full training stack. Exact architectural and evaluation differences are documented in `docs/METHOD.md` and should be preserved in any publication.
