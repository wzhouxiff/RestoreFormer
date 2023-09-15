# Updating
- **20230915** Update an online demo [![Huggingface Gradio](https://img.shields.io/static/v1?label=Demo&message=Huggingface%20Gradio&color=orange)](https://huggingface.co/spaces/wzhouxiff/RestoreFormerPlusPlus)
- **20230915** A more user-friendly and comprehensive inference method refer to our [RestoreFormer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus)
- **20230116** For convenience, we further upload the [test datasets](#testset), including CelebA (both HQ and LQ data), LFW-Test, CelebChild-Test, and Webphoto-Test, to OneDrive and BaiduYun.
- **20221003** We provide the link of the [test datasets](#testset).
- **20220924** We add the code for [**metrics**](#metrics) in scripts/metrics.


# RestoreFormer

This repo includes the source code of the paper: "[RestoreFormer: High-Quality Blind Face Restoration from Undegraded Key-Value Pairs](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_RestoreFormer_High-Quality_Blind_Face_Restoration_From_Undegraded_Key-Value_Pairs_CVPR_2022_paper.pdf)" (CVPR 2022) by Zhouxia Wang, Jiawei Zhang, Runjian Chen, Wenping Wang, and Ping Luo.

![](assets/figure1.png)

**RestoreFormer** tends to explore fully-spatial attentions to model contextual information and surpasses existing works that use local operators. It has several benefits compared to prior arts. First, it incorporates a multi-head coross-attention layer to learn fully-spatial interations between corrupted queries and high-quality key-value pairs. Second, the key-value pairs in RestoreFormer are sampled from a reconstruction-oriented high-quality dictionary, whose elements are rich in high-quality facial features specifically aimed for face reconstruction.

<!-- ![](assets/framework.png "Framework")-->

## Environment

- python>=3.7
- pytorch>=1.7.1
- pytorch-lightning==1.0.8
- omegaconf==2.0.0
- basicsr==1.3.3.4

**Warning** Different versions of pytorch-lightning and omegaconf may lead to errors or different results.

## Preparations of dataset and models

**Dataset**: 
- Training data: Both **HQ Dictionary** and **RestoreFormer** in our work are trained with **FFHQ** which attained from [FFHQ repository](https://github.com/NVlabs/ffhq-dataset). The original size of the images in FFHQ are 1024x1024. We resize them to 512x512 with bilinear interpolation in our work. Link this dataset to ./data/FFHQ/image512x512.
- <a id="testset">Test data</a>: <!--[CelebA-Test](https://pan.baidu.com/s/1iUvBBFMkjgPcWrhZlZY2og?pwd=test), [LFW-Test](http://vis-www.cs.umass.edu/lfw/#views), [WebPhoto-Test](https://xinntao.github.io/projects/gfpgan), and [CelebChild-Test](https://xinntao.github.io/projects/gfpgan)-->
   * CelebA-Test-HQ: [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/EY7P-MReZUZOngy3UGa5abUBJKel1IH5uYZLdwp2e2KvUw?e=rK0VWh); [BaiduYun](https://pan.baidu.com/s/1tMpxz8lIW50U8h00047GIw?pwd=mp9t)(code mp9t)
   * CelebA-Test-LQ: [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/EXULDOtX3qdKg9_--k-hbr4BumxOUAi19iQjZNz75S6pKA?e=Kghqri); [BaiduYun](https://pan.baidu.com/s/1y6ZcQPCLyggj9VB5MgoWyg?pwd=7s6h)(code 7s6h)
   * LFW-Test: [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/EZ7ibkhUuRxBjdd-MesczpgBfpLVfv-9uYVskLuZiYpBsg?e=xPNH26); [BaiduYun](https://pan.baidu.com/s/1UkfYLTViL8XVdZ-Ej-2G9g?pwd=7fhr)(code 7fhr). Note that it was align with dlib.
   * CelebChild: [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/ESK6vjLzDuJAsd-cfWrfl20BTeSD_w4uRNJREGfl3zGzJg?e=Tou7ft); [BaiduYun](https://pan.baidu.com/s/1pGCD4TkhtDsmp8emZd8smA?pwd=rq65)(code rq65)
   * WepPhoto-Test: [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/ER1-0eYKGkZIs-YEDhNW0xIBohCI5IEZyAS2PAvI81Stcg?e=TFJFGh); [BaiduYun](https://pan.baidu.com/s/1SjBfinSL1F-bbOpXiD0nlw?pwd=nren)(code nren)

**Model**: Both pretrained models used for training and the trained model of our RestoreFormer can be attained from [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wzhoux_connect_hku_hk/Eb73S2jXZIxNrrOFRnFKu2MBTe7kl4cMYYwwiudAmDNwYg?e=Xa4ZDf) or [BaiduYun](https://pan.baidu.com/s/1EO7_1dYyCuORpPNosQgogg?pwd=x6nn)(code x6nn). Link these models to ./experiments.

## Test
    sh scripts/test.sh

## Training
    sh scripts/run.sh

**Note**. 
- The first stage is to attain **HQ Dictionary** by setting `conf_name` in scripts/run.sh to 'HQ\_Dictionary'. 
- The second stage is blind face restoration. You need to add your trained HQ\_Dictionary model to `ckpt_path` in config/RestoreFormer.yaml and set `conf_name` in scripts/run.sh to 'RestoreFormer'.
- Our model is trained with 4 V100 GPUs.

## <a id="metrics">Metrics</a>
    sh scripts/metrics/run.sh
    
**Note**. 
- You need to add the path of CelebA-Test dataset in the script if you want get IDD, PSRN, SSIM, LIPIS.

## Citation
    @article{wang2022restoreformer,
      title={RestoreFormer: High-Quality Blind Face Restoration from Undegraded Key-Value Pairs},
      author={Wang, Zhouxia and Zhang, Jiawei and Chen, Runjian and Wang, Wenping and Luo, Ping},
      booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2022}
    }

## Acknowledgement
We thank everyone who makes their code and models available, especially [Taming Transformer](https://github.com/CompVis/taming-transformers), [basicsr](https://github.com/XPixelGroup/BasicSR), and [GFPGAN](https://github.com/TencentARC/GFPGAN).

## Contact
For any question, feel free to email `wzhoux@connect.hku.hk` or `zhouzi1212@gmail.com`.
