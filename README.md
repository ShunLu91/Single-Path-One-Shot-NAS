# Single-Path-One-Shot-NAS
![license](https://img.shields.io/badge/License-MIT-brightgreen)
![python](https://img.shields.io/badge/Python-3.7-blue)
![pytorch](https://img.shields.io/badge/PyTorch-1.7-orange)

This repo provides a Pytorch-based implementation of SPOS([Single Path One-Shot Neural Architecture Search with Uniform 
Sampling](https://arxiv.org/abs/1904.00420))  by Zichao Guo, and et. al.
![SPOS](./img/SPOS.jpg)

It only contains 'Block Search' for reference. It's very time consuming to train this supernet on ImageNet, which
makes it impossible for me to finish the experiment under limited resources. Therefore, I mainly focused on 
CIFAR-10. 

Great thanks to Zichao Guo for his advice on some details. Nevertheless, some differences may still exists when compared with the [official version](https://github.com/megvii-model/SinglePathOneShot) 
such as data preprocessing and some other hyper parameters.

## Environments    
```
Python==3.7.10, Pytorch==1.7.1, CUDA==10.2, cuDNN==7.6.5 
```

## Dataset   
CIFAR-10 can be automatically downloaded using this code. ImageNet needs to be manually downloaded and 
[here](https://github.com/pytorch/examples/tree/master/imagenet) are some instructions. 
         
## Usage
1. Train a supernet on the CIFAR-10 dataset by simply running:
```
bash scripts/train_supernet.sh
```
* My pretrained supernet can be downloaded from [this link](https://drive.google.com/file/d/1hq3uaCqHnIL_foD-aeieqD_HUxj1xPME/view?usp=sharing).

2. For convenience, I conduct random search by enumerating 1,000 paths to select the best:
```
bash scripts/random_search.sh
```
* During my search, the best path is `[1, 0, 3, 1, 3, 0, 3, 0, 0, 3, 3, 0, 1, 0, 1, 2, 2, 1, 1, 3]`.
* In the original SPOS paper, they adopted the evolutionary algorithm to search architectures. Please refer to their official repo for more details.

3. Use the best searched path to modify the "choice" defined in Line 116 of retrain_best_choice.py and re-train the corresponding architecture of this path:
```
bash scripts/retrain_best_choice.py
```
* After retraining, the best test accuracy of this searched architecture is `93.31`. The checkpoint is provided [here](https://drive.google.com/file/d/1Ld7wBaZd7ikeOolXdncLgk0lIw2WHo0d/view?usp=sharing).


4. As I fix all seeds in the above procedures, same results should be achieved. You can check my logs in the `logdir`. 

## Reference
[1] [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://github.com/megvii-model/SinglePathOneShot)

[2] [Differentiable architecture search for convolutional and recurrent networks](https://github.com/quark0/darts)
