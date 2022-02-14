# 基于深度能量模型的低剂量CT重建

## 摘要
降低计算机断层扫描（CT）的剂量对于降低临床应用中的辐射风险至关重要。深度学习的快速发展和广泛应用为低剂量CT成像算法的发展带来了新的方向。与大多数受益于手动设计的先验函数或监督学习方案的现有先验驱动算法不同，本文使用基于深度能量模型来学习正常剂量CT的先验知识，然后在迭代重建阶段，将数据一致性作为条件项集成到低剂量CT的迭代生成模型中，通过郎之万动力学迭代更新训练的先验。实验比较并证明了所提方法的降噪和细节保留能力优良。

## Requirements and Dependencies
    scipy==0.19.1
    horovod==0.16.0
    torch==0.3.1
    scikit_image==0.13.0
    tensorflow==1.12.0
    torchvision==0.2.0
    six==1.11.0
    imageio==2.4.1
    tqdm==4.20.0
    matplotlib==1.5.3
    mpi4py==3.0.0
    numpy==1.14.0
    Pillow==5.4.1
    baselines==0.1.5
    skimage==0.0
    scikit_learn==0.20.3
    ODL==1.0.0dev
    astra-toolbox==2.0.0
    CUDA==10.0

## Test
    python CT_test6.py --exp=CT512_lr3e-5 --resume_iter=45000 --swish_act --step_lr 10 --num_steps 2000

## Checkpoints
The pretrained checkpoints can be download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1tbkTcSxQ8KhD3ILAAmBXjg ). 
key number is "ebm1" 


## Other Related Projects
  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)




