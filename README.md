# NeRDF
> Efficient View Synthesis with Neural Radiance Distribution Field <br />
> [GAP Lab](https://gaplab.cuhk.edu.cn/), [Yushuang Wu](https://scholar.google.com/citations?hl=zh-CN&user=x5gpN0sAAAAJ)

![Teaser](figures/teaser_2k.png)

[Paper](https://arxiv.org/abs/2304.10179.pdf) - 
[Project Website](https://yushuang-wu.github.io/SCoDA/) -
[Arxiv](https://arxiv.org/abs/2304.10179) -
Published in ICCV 2023.

#### Citation

If you find our code or paper useful for your project, please consider citing:

    @inproceedings{wu2023nerdf,
      title={Efficient View Synthesis with Neural Radiance Distribution Field},
      author={Yushuang, Wu and Xiao, Li and Shuguang, Cui and Xiaoguang, Han and Yan, Lu},
      booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
      year={2023},
    }
    
## NeRDF



![Dataset](figures/dataset_vis.png)

> ScanSalon Data: At [Google Drive](https://drive.google.com/drive/folders/1JrBxlBufivinI5_Xyi-1wBz2quU-DThj?usp=sharing) (Real data only).  <br />
> ShapeNet Data: Turn to [ShapeNet](https://shapenet.org) for synthetic data download.  <br />
> Processing: Please refer to the README and scripts in the zip package.
    
## Installation

Our implementation is based on IF-Net as the basic framework for reconstruction. Please refer to the "Install" part of [IF-Net](https://github.com/jchibane/if-net) for the installation of our method. 

## Running

1. Following the steps in data_processing/mesh-fusion to get the water-tight ScanSalon meshes. <br />
2. Following the steps in [Mesh2PC](https://github.com/kochanha/Mesh-to-Pointcloud-using-Blensor) to get simulated scans from ShapeNet meshes. <br /> 
3. Following the steps in data_processing/process.sh to preprocess all data. <br />
4. Run `train_ddp.sh` to train the model in a parrallel way. <br />
5. After training by around 30-50 epochs, run `generate_ddp.sh` to generate meshes in the test set. 

![Methodology](figures/method.png)
