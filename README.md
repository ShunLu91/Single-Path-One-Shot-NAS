# Single-Path-One-Shot-NAS
This repo provides a Pytorch-based implementation of SPOS([Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420))  by Zichao Guo, and et. al.
![SPOS](https://github.com/ShunLu91/Single-Path-One-Shot-NAS/blob/master/img/SPOS.jpg)
However, this repo only contains 'Block Search' and it's very time consuming to train this network on ImageNet so I haven't got the final result yet. I will update the repo soon and greatly thanks to Zichao Guo for his advice on some details.      
Yet, there are still some differences with the [official version](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot) such as data preprocessing and some hyper parameters. Nevertheless, I'd appreciate it if you find some bugs in this repo or get the final result because I won't have so much time to check this for the starting of school. Welcome to pull requests here.        
                
## Environments    
```
Python == 3.6.8, Pytorch == 1.1.0, CUDA == 9.0.176, cuDNN == 7.3.0, GPU == Single GTX 1080Ti 
```

## Dataset   
SPOS directly can train on ImageNet.ImageNet needs to be manually downloaded and [here](https://github.com/pytorch/examples/tree/master/imagenet) are some instructions.   
         
## Usage
```
python train.py --train_dir YOUR_TRAINDATASET_PATH --val_dir YOUR_VALDATASET_PATH
```

## To Do
- [x] Block Search
- [ ] Channel Search
- [ ] Evolutionary Algorithm
             
## Citation
```Python
@article{guo2019single,
        title={Single path one-shot neural architecture search with uniform sampling},
        author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
        journal={arXiv preprint arXiv:1904.00420},
        year={2019}
}
```
