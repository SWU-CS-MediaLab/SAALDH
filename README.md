# Self-Attention and Adversary Learning Deep Hashing Network for Cross-modal Retrieval (SAALDH)
## Abstract

Multi-modal information retrieval is among the prevailing forms of daily human–computer interaction. The recent deep cross-modal hashing methods have received increasing attention
because of their superior search performance and efficiency capability. However, effectively exploring the high-ranking semantic correlation and preserving representation consistency are still challengeable due to the heterogeneity property of different modalities. In this paper, a Self-Attention and Adversary Learning Hashing Network (SAALDH) is designed for large scale cross-modal retrieval. Specifically, the hash representations across different layers of the deep network are integrated and then the significance of each position in the integrated hash representation is enhanced by employing a novel self-attention mechanism. Meanwhile, an adversarial learning mechanism is adopted to further preserve the consistency of hash representations during hash functions learning. Moreover, a novel batch semi-hard selection is designed for triplet loss to solve the issue of local optimum during the optimization of SAALDH. Experimental results evaluated on two large scale image-text modality datasets show the effectiveness and efficiency of the proposed SAALDH. And SAALDH achieves better performances by comparing with several state-of-the-art methods. 

------

Please cite our paper if you use this code in your own work:

@article{SAALDH,
title = {Self-attention and adversary learning deep hashing network for cross-modal retrieval},
journal = {Computers & Electrical Engineering},
volume = {93},
pages = {107262},
year = {2021},
issn = {0045-7906},
doi = {https://doi.org/10.1016/j.compeleceng.2021.107262},
url = {https://www.sciencedirect.com/science/article/pii/S0045790621002457},
author = {Shubai Chen and Song Wu and Li Wang and Zhenyang Yu}
}

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----
