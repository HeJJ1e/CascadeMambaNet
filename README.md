# CascadeMambaNet

CascadeMambaNet is a four-stage cascade network for semantic segmentation of building-facade point clouds. This repository provides the official implementation and the corresponding data pipelines used for experiments on the BF and ZAHA datasets.

The release keeps one model implementation for both datasets. Dataset-specific behavior is selected with `--dataset bf` or `--dataset zaha`; no source file needs to be replaced.

## Model Configuration

| Setting | Value |
| --- | --- |
| Input points | 40,960 |
| Batch size | 4 |
| Stage points | 2,560 / 5,120 / 20,480 / 40,960 |
| Stage dimensions | 160 / 160 / 176 / 176 |
| Stage blocks | 3 / 3 / 4 / 4 |
| Neighborhood size | 16 |
| Training epochs | 200 |
| Optimizer | AdamW |
| Learning rate | 0.0002 |
| Weight decay | 0.0005 |
| EMA decay | 0.999 |

The fixed model includes GADPE, three-order point serialization, bidirectional Mamba blocks with channel state-space modeling, KNN cascade propagation, multi-stage gated refinement, uncertainty-guided class refinement, and full-scale pyramid aggregation. The training objective uses class-balanced focal loss, Lovasz-Softmax loss, boundary weighting, contrastive boundary learning, and four-stage supervision.

## Dataset Differences

| Setting | BF | ZAHA |
| --- | --- | --- |
| Classes | 8 | 15 |
| Grid size | 0.04 m | 0.20 m |
| RGB | Used | Not used |
| Input features | XYZ + RGB + normal + 5D geometry | XYZ + normal + 5D geometry |
| Serialization grid | 0.02 m | 0.10 m |
| Stage grids | 0.16 / 0.12 / 0.06 / 0.04 m | 0.80 / 0.60 / 0.30 / 0.20 m |

Both datasets use 40,960 sampled points and the same stage capacities. The dataset-specific values are defined in `configs/settings.py`.

## Datasets

- BF Building Facade dataset: [Google Drive](https://drive.google.com/drive/folders/1cZEUnyF3jn0UnQNrhlZCkVjb54XzRTBd?hl=en)
- ZAHA dataset: [official repository](https://github.com/OloOcki/zaha) and [official download page](https://tum2t.win/datasets/pc-mls)

The datasets are not redistributed in this repository. Arrange the downloaded data as follows:

```text
CascadeMambaNet/
  data/
    BF/
      ZHC_Building_Facade/
        Area_1/
        ...
        Area_6/
    ZAHA/
      raw_pcd/
```

## Environment

The reference experiments used Ubuntu Linux, Python 3.9.25, PyTorch 2.5.1 with CUDA 11.8, Mamba-SSM 2.3.1, causal-conv1d 1.6.1, NumPy 1.26.4, SciPy 1.13.1, and scikit-learn 0.24.2.

```bash
conda create -n cascademambanet python=3.9 -y
conda activate cascademambanet
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install causal-conv1d==1.6.1 --no-build-isolation
pip install mamba-ssm==2.3.1 --no-build-isolation
```

CUDA extension installation depends on the local compiler and CUDA toolkit. Verify the Mamba and causal-conv1d CUDA forward/backward passes before training.

## Preprocessing

Prepare BF:

```bash
python -m datasets.bf.preprocess
```

Prepare ZAHA:

```bash
python -m datasets.zaha.preprocess
```

The preprocessing scripts create subsampled point clouds, KD-trees, full-cloud projections, normals, and geometric descriptors.

## Training

BF Area 5:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python run.py --dataset bf --gpu 0 --test_area 5 --mode train
```

ZAHA:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python run.py --dataset zaha --gpu 0 --mode train
```

## Evaluation

BF Area 5:

```bash
CUDA_VISIBLE_DEVICES=0 \
python run.py --dataset bf --gpu 0 --test_area 5 --mode test \
  --model-path checkpoints/CascadeMambaNet_BF_Area5.pth
```

ZAHA official test split:

```bash
CUDA_VISIBLE_DEVICES=0 \
python run.py --dataset zaha --gpu 0 --mode test \
  --model-path checkpoints/CascadeMambaNet_ZAHA.pth
```

BF evaluation writes full-cloud prediction PLY files and a test log. ZAHA evaluation writes prediction PLY files and `metrics.json` with OA, mean precision, mean recall, mean F1, mean IoU, per-class scores, and the confusion matrix. Test-time augmentation is disabled.

## Checkpoints

Pretrained weights are available from the [CascadeMambaNet v1.0.0 release](https://github.com/HeJJ1e/CascadeMambaNet/releases/tag/v1.0.0).

| Dataset | Checkpoint |
| --- | --- |
| BF Area 5 | [CascadeMambaNet_BF_Area5.pth](https://github.com/HeJJ1e/CascadeMambaNet/releases/download/v1.0.0/CascadeMambaNet_BF_Area5.pth) |
| ZAHA | [CascadeMambaNet_ZAHA.pth](https://github.com/HeJJ1e/CascadeMambaNet/releases/download/v1.0.0/CascadeMambaNet_ZAHA.pth) |

Place the downloaded files in `checkpoints/`. The checkpoints are plain model `state_dict` files and are loaded with strict key matching.

## References

Su, Y.; Liu, W.; Yuan, Z.; Cheng, M.; Zhang, Z.; Shen, X.; Wang, C. DLA-Net: Learning Dual Local Attention Features for Semantic Segmentation of Large-Scale Building Facade Point Clouds. Pattern Recognition 2022, 123, 108372. [https://doi.org/10.1016/j.patcog.2021.108372](https://doi.org/10.1016/j.patcog.2021.108372).

Gu, A.; Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv 2023, arXiv:2312.00752. [https://doi.org/10.48550/arXiv.2312.00752](https://doi.org/10.48550/arXiv.2312.00752).

Zhang, T.; Yuan, H.; Qi, L.; Zhang, J.; Zhou, Q.; Ji, S.; Yan, S.; Li, X. Point Cloud Mamba: Point Cloud Learning via State Space Model. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025, 39(10), pp. 10121–10130. [https://doi.org/10.1609/aaai.v39i10.33098](https://doi.org/10.1609/aaai.v39i10.33098).
