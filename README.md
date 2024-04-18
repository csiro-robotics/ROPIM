# ROPIM: Pre-training with Random Orthogonal Projection Image Modeling
ROPIM is a self-supervised learning technique based on count sketching, which  reduces local semantic information under the bounded noise variance. While  Masked Image Modelling (MIM) introduces Binary noise, ROPIM proposes a _continous_ masking strategy. 
Continuous masking allows for larger number of masking patterns compared to binary masking.
![alt text](figures/ROPIM.png)


## Citation

If you find our work useful for your research, please consider giving a star :star: and citation :beer::

```bibtex
@inproceedings{
haghighat2024pretraining,
title={Pre-training with Random Orthogonal Projection Image Modeling},
author={Maryam Haghighat and Peyman Moghadam and Shaheer Mohamed and Piotr Koniusz},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=z4Hcegjzph}
}
```

## Acknowledgement

This code is built using the [timm](https://github.com/huggingface/pytorch-image-models) library, the [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repository and the [SimMIM](https://github.com/microsoft/SimMIM) repository.
