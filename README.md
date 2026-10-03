# AcTOL: Vision-Language Pretraining from Human Videos for Robot Learning

**Action Temporal Coherence Learning (AcTOL)** learns ordered and continuous vision-language representations from human action videos for robot learning and language-conditioned manipulation.

Official implementation of **[Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](https://papers.neurips.cc/paper_files/paper/2025/hash/68fa5b1f1c024639685be9668fd5dc32-Abstract-Conference.html)** (NeurIPS 2025, Poster).  
Zhizhen Zhang, Lei Zhu, Zhen Fang, Zi Huang, and Yadan Luo.

[Paper](https://papers.neurips.cc/paper_files/paper/2025/hash/68fa5b1f1c024639685be9668fd5dc32-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2502.01218) · [Project page](https://actol-pretrain.github.io/) · [Pretrained checkpoint](#model-zoo) · [Citation](#citation)

## Overview

AcTOL studies temporal representation learning for robotics through vision-language pretraining on human videos. It combines an ordering objective with a local Brownian bridge constraint to model temporal continuity, without assuming that the final video frame always represents the language-specified goal.

- **Pretraining data:** human action videos and language descriptions from EPIC-KITCHENS-100.
- **Representation:** visual and language features that capture action semantics, temporal ordering, and continuity.
- **Downstream evaluation:** language-conditioned behavior cloning for simulated and real-world robot manipulation, together with visual reward analysis.

## Abstract
Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments on both simulated and real robots show that the pretrained features significantly enhance downstream manipulation tasks with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents.

![Demo](assets/demo.gif)

## Installation

### Prerequisites
- **Python 3.8+**
- **PyTorch 1.13.1+**


### Install AcTOL
```bash
pip install -e .
```

---

## Training AcTOL

To train AcTOL on **EPIC-KITCHENS-100**, run:

```bash
python main.py --image_path /path/to/data \
               --meta_file assets/EpicKitchen-100-train.csv \
               --batch-size 128 --epochs 1001 \
               --input-size 224 --num_frames 10 \
               --model RN50 --vlo_temp 0.01 \
               --opt adam --lr 1e-5 \
               --output_dir ./checkpoints
```

### Key Arguments:
| Argument         | Description |
|-----------------|-------------|
| `--image_path`  | Path to the dataset directory. |
| `--meta_file`   | Path to metadata CSV file. |
| `--batch-size`  | Number of videos per batch. |
| `--num_frames`  | Number of sampled frames per video. |
| `--model`       | CLIP model backbone (`RN50`). |
| `--vlo_temp`    | Temperature for VLO loss. |
| `--lr`          | Learning rate. |
| `--output_dir`  | Directory to save checkpoints. |

---

## Model Zoo
| Models    | Pretraining Methods | Params<br />(M) | Epochs | Pretrain ckpt                                                                              |
| --------- | ------------------- | --------------- | ----- | ------------------------------------------------------------------------------------------ |
| RN50-CLIP | AcTOL       | 386             | 1000    | [link](https://drive.google.com/file/d/19GX5k0CjjHoCqhTSwNdmAqiNBlNnuiVw/view?usp=sharing) |


## Evaluation

To evaluate the language conditioned visual reward:
```python

import AcTOL
import torch
from PIL import Image
# Load AcTOL model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AcTOL.load("AcTOL", device=device)

image = Image.open("Your Image Path Here")
text = "Your Instruction Here"

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    reward = model.get_reward(image, text)
```
To evaluate the language-conditioned behavior cloning, please refer to the evaluation code of [R3M](https://github.com/facebookresearch/r3m/tree/eval/evaluation).

---

## Citation
Kindly cite our paper if you find it helpful:
```bibtex
@inproceedings{zhang2025actol,
  author       = {Zhizhen Zhang and Lei Zhu and Zhen Fang and Zi Huang and Yadan Luo},
  title        = {Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents},
  booktitle    = {Advances in Neural Information Processing Systems},
  volume       = {38},
  year         = {2025},
  doi          = {10.52202/085713-2428},
  url          = {https://papers.neurips.cc/paper_files/paper/2025/hash/68fa5b1f1c024639685be9668fd5dc32-Abstract-Conference.html}
}
```
## Acknowledgements
Part of this code are adapted from [DecisionNCE](https://github.com/2toinf/DecisionNCE.git) and [RnC](https://github.com/kaiwenzha/Rank-N-Contrast.git), thanks for their excellent work!
