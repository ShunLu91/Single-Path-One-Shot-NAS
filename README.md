# Single-Path-One-Shot-NAS
This repo provides a Pytorch-based implementation of SPOS([Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420))  by Zichao Guo, and et. al.
![SPOS](https://github.com/ShunLu91/Single-Path-One-Shot-NAS/blob/master/img/SPOS.jpg)
However,this repo only contains 'Block Search' and it's very time consuming to train this Network on ImageNet so I haven't got the final result yet.I will update the repo soon.      
I'd appreciate it if you find some bugs in this code or get the final result because I won't have so much time to check this for the starting of school. Please contact me at 470651748@qq.com.        
                
## Environments:    
```Python
Python == 3.6.8, Pytorch == 1.1.0, CUDA == 9.0.176, cuDNN == 7.3.0, GPU == GTX 1080 Ti 
```

## Dataset:   
SPOS directly can train on ImageNet.ImageNet needs to be manually downloaded and [here](https://github.com/pytorch/examples/tree/master/imagenet) are some instructions.   

## To Do:
- [x] Block Search
- [ ] Channel Search
- [ ] Evolutionary Algorithm

