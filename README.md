# AcTOL: Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents

This repository provides the **official implementation** of the ICML 2025 submission paper:  
**"Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents"**.

## 🔥Abstract
Pre-training vision-language representations on human action videos has emerged as a promising approach to reduce reliance on large-scale expert demonstrations for training embodied agents. However, prior methods often employ time contrastive learning based on goal-reaching heuristics, progressively aligning language instructions from the initial to the final frame. This overemphasis on future frames can result in erroneous vision-language associations, as actions may terminate early or include irrelevant moments in the end. To address this issue, we propose Action Temporal Coherence Learning (AcTOL) to learn ordered and continuous vision-language representations without rigid goal-based constraint. AcTOL treats a video as a continuous trajectory where it (1) contrasts semantic differences between frames to reflect their natural ordering, and (2) imposes a local Brownian bridge constraint to ensure smooth transitions across intermediate frames. Extensive imitation learning experiments across varying numbers of demonstrations show that the pretrained features significantly enhance downstream manipulation tasks with high robustness to different linguistic styles of instructions, offering a viable pathway toward generalized embodied agents.


## 🚀 Installation

### Prerequisites
- **Python 3.8+**
- **PyTorch 1.13.1+**

### Install Dependencies
```bash
pip install -r requirements.txt
```
or install manually:
```bash
pip install torch==1.13.1 torchvision==0.14.1 timm==0.9.12 mmengine tqdm numpy tensorboardX gdown openai-clip chardet
```

### Install AcTOL
```bash
pip install -e .
```

---

## 🎯 Training AcTOL

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
| `--batch-size`  | Training batch size. |
| `--epochs`      | Number of training epochs. |
| `--model`       | CLIP model backbone (`RN50`). |
| `--vlo_temp`    | Temperature for VLO loss. |
| `--lr`          | Learning rate. |
| `--output_dir`  | Directory to save checkpoints. |

---

## 📊 Evaluation

To evaluate the trained model:
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

---
