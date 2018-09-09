## PSANet: Point-wise Spatial Attention Network for Scene Parsing (in construction)

by Hengshuang Zhao\*, Yi Zhang\*, Shu Liu, Jianping Shi, Chen Change Loy, Dahua Lin, Jiaya Jia, details are in [project page](https://hszhao.github.io/projects/psanet).

### Introduction

This repository is build for PSANet, which contains source code for PSA module and related evaluation code. For installation, please merge the related layers and follow the description in [PSPNet](https://github.com/hszhao/PSPNet) repository (test with CUDA 7.0/7.5 + cuDNN v4).

**Note**: There are some small structure changes of the evaluation code compared to PSPNet, and has not been fully tested yet for limited time, and PSPNet evaluation should work well if you prefer . After the coming ECCV2018, both PSPNet and PSANet will be updated to CUDA 8.0 + cuDNN v5, both docker file and python interface scripts will be provided, and pytorch implementation will be available in repo [semseg](https://github.com/hszhao/semseg) later.

### Usage

1. Clone the repository recursively:

   ```shell
   git clone --recursive https://github.com/hszhao/PSANet.git
   ```

2. Merge the caffe layers into PSPNet repository:

   Point-wise spatial attention: pointwise_spatial_attention_layer.hpp/cpp/cu and caffe.proto.

3. Build Caffe and matcaffe:

   ```shell
   cd $PSANET_ROOT/PSPNet
   cp Makefile.config.example Makefile.config
   vim Makefile.config
   make -j8 && make matcaffe
   cd ..
   ```

4. Evaluation:

   - Evaluation code is in folder 'evaluation'.
   - Download trained models and put them in related dataset folder under 'evaluation/model', refer '[README.md](evaluation/model/README.md)'.
   - Modify the related paths in 'eval_all.m':

     Mainly variables 'data_root' and 'eval_list', and your image list for evaluation should be similarity to that in folder 'evaluation/samplelist' if you use this evaluation code structure.

   ```shell
   cd evaluation
   vim eval_all.m
   ```

   - Run the evaluation scripts:

   ```
   ./run.sh
   ```

5. Results: 

   Predictions will show in folder 'evaluation/mc_result' and the expected scores are listed as below:

   (mIoU/pAcc. stands for mean IoU and pixel accuracy, 'ss' and 'ms' denote single scale and multiple scale testing.)

   ADE20K:

   |  network  | training data | testing data | mIoU/pAcc.(ss) | mIoU/pAcc.(ms) |                            md5sum                            |
   | :-------: | :-----------: | :----------: | :------------: | :------------: | :----------------------------------------------------------: |
   | PSANet50  |     train     |     val      |  41.92/80.17   |  42.97/80.92   | [a8e884](https://drive.google.com/file/d/1F1A-ddhhppAQxSaTRWgIlQL8NMa4VMLV/view?usp=sharing) |
   | PSANet101 |     train     |     val      |  42.75/80.71   |  43.77/81.51   | [ab5e56](https://drive.google.com/file/d/1u8ntKfkNgxmrBjH3U_3zbGKvLndpxwtk/view?usp=sharing) |

   VOC2012:

   |  network  |     training data      | testing data | mIoU/pAcc.(ss) | mIoU/pAcc.(ms) |                            md5sum                            |
   | :-------: | :--------------------: | :----------: | :------------: | :------------: | :----------------------------------------------------------: |
   | PSANet50  |       train_aug        |     val      |  77.24/94.88   |  78.14/95.12   | [d5fc37](https://drive.google.com/file/d/1uZLdv-1ReOJuRau06VEib0FOUb0I-fpl/view?usp=sharing) |
   | PSANet101 |       train_aug        |     val      |  78.51/95.18   |  79.77/95.43   | [5d8c0f](https://drive.google.com/file/d/11dGNxh4nzoiV4fscJPcRD-OG9YsKSoaI/view?usp=sharing) |
   | PSANet101 | COCO + train_aug + val |     test     |      -/-       |     85.7/-     | [3c6a69](https://drive.google.com/file/d/19sBwiQJh3pOj9LoFGhMnzpBA-pmDtmP3/view?usp=sharing) |

   Cityscapes:

   |  network  |     training data     | testing data | mIoU/pAcc.(ss) | mIoU/pAcc.(ms) |                            md5sum                            |
   | :-------: | :-------------------: | :----------: | :------------: | :------------: | :----------------------------------------------------------: |
   | PSANet50  |      fine_train       |   fine_val   |  76.65/95.99   |  77.79/96.24   | [25c06a](https://drive.google.com/file/d/1nr73jW42eWf5Xy1_Ch1RwpjwC4f5tCUk/view?usp=sharing) |
   | PSANet101 |      fine_train       |   fine_val   |  77.94/96.10   |  79.05/96.30   | [3ac1bf](https://drive.google.com/file/d/1uaNZl7HgqYWwtPsKVREKIoo7Ib9jXxB2/view?usp=sharing) |
   | PSANet101 |      fine_train       |  fine_test   |      -/-       |     78.6/-     | [3ac1bf](https://drive.google.com/file/d/1uaNZl7HgqYWwtPsKVREKIoo7Ib9jXxB2/view?usp=sharing) |
   | PSANet101 | fine_train + fine_val |  fine_test   |      -/-       |     80.1/-     | [1dfc91](https://drive.google.com/file/d/1ZUT8g_Lx5Iih4lkZk3meAC6dpNk-ZJxT/view?usp=sharing) |

6. Demo video:

   - Video processed by PSANet (with PSPNet) on [BDD](http://bdd-data.berkeley.edu) dataset for drivable area segmentation: [Video](https://youtu.be/l5xu1DI6pDk).

### Citation

If PSANet is useful for your research, please consider citing:

    @inproceedings{zhao2018psanet,
      title={{PSANet}: Point-wise Spatial Attention Network for Scene Parsing},
      author={Zhao, Hengshuang and Zhang, Yi and Liu, Shu and Shi, Jianping and Loy, Chen Change and Lin, Dahua and Jia, Jiaya},
      booktitle={ECCV},
      year={2018}
    }

### Questions

Please contact 'hszhao@cse.cuhk.edu.hk' or 'zy217@ie.cuhk.edu.hk'
