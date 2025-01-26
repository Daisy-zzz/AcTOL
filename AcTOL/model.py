import clip
import gdown
import torch
import os
import torch.nn as nn
import torchvision.transforms as T
from typing import Union
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class CLIPBasedEncoder(nn.Module):
    def __init__(self, modelid="RN50", device="cuda"):
        
        super().__init__()
        self.modelid = modelid
        self.device = device
        # Load CLIP model and transform
        model, cliptransforms = clip.load(modelid, device=self.device, jit=False)
        # CLIP precision
        model.float()
        self.model = model 
        del self.model.logit_scale
        self.model.train()
        self.transforms = cliptransforms
        self.transforms_tensor = nn.Sequential(
                T.Resize(self.model.visual.input_resolution, interpolation=BICUBIC,antialias=None),
                T.CenterCrop(self.model.visual.input_resolution),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            )
        # self.mapping_mlp = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024)
        # )
        # self.freeze_text()
        # self.freeze_all_except_last()
    def freeze_text(self):
        for param in self.model.transformer.parameters():
            param.requires_grad = False
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False
        self.model.positional_embedding.requires_grad = False
        for param in self.model.ln_final.parameters():
            param.requires_grad = False
        self.model.text_projection.requires_grad = False
            
    def freeze_all_except_last(self):
        # Freeze all layers except the last layer in the visual model
        for name, param in self.model.visual.named_parameters():
            if 'layer4' not in name and 'attnpool' not in name:
                param.requires_grad = False
    
    def get_reward(self, visual_input, text_input):
        visual_feature = self.encode_image(visual_input)
        text_feature = self.encode_text(text_input) 
        return torch.nn.functional.cosine_similarity(visual_feature, text_feature, dim=-1)


    def encode_image(self, visual_input):
        if type(visual_input) != torch.Tensor:
            visual_input = self.transforms(visual_input).to(self.device)
            if len(visual_input.shape) == 3: visual_input = visual_input.unsqueeze(0)
        else:
            if torch.max(visual_input) > 10.0:
                visual_input = visual_input / 255.0
            visual_input = self.transforms_tensor(visual_input).to(self.device)

        return self.model.encode_image(visual_input)
        
    def encode_text(self, text_input):
        # with torch.no_grad():
        if type(text_input) == str:
            text_input = [text_input]
        if type(text_input) != torch.Tensor:
            text_input = clip.tokenize(text_input).to(self.device)
        return self.model.encode_text(text_input)
    
    def forward(self, visual_input, text_input):
        return self.encode_image(visual_input), self.encode_text(text_input)

    # def lang_cond(self, input):
    #     return self.mapping_mlp(input)
    
_MODELS = {
    "vit":
        {
            "modelid": "ViT-B/32",
            "download_link": "https://drive.google.com/uc?export=download&id=1LmDHaKMZCv9QT89dWubZ8dRo6qwpVMYo",
        }
}

# P: https://drive.google.com/file/d/1LmDHaKMZCv9QT89dWubZ8dRo6qwpVMYo/view?usp=drive_link
# T: https://drive.google.com/file/d/14wn2R5ZDNujSq9Tsaeuy6fJNr6zM5l7E/view?usp=drive_link

def _download(url: str, name: str,root: str):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, name)
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
    if os.path.isfile(download_target):
        return download_target
    gdown.download(url, download_target, quiet=False)
    return download_target

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    
    if name == "ViT-B/32":
        model_path = "/root/.cache/Project/ckpt_20ep.pth"
        print(model_path)
    elif name == "rncbb":
        model_path = "/data/guohua/BeiJing/zzz/RobotPVR/Project/runnings/RnC/rncbb.pth"
    elif name == "rnc":
        model_path = "/data/guohua/BeiJing/zzz/RobotPVR/Project/runnings/RnC/rncf10.pth"
    else:
        raise RuntimeError(f"Model {name} not found; available models = {_MODELS.keys()}")
    print(f"===========Loading '{model_path}' Model==============")
    model = CLIPBasedEncoder("RN50", device)
    with open(model_path, 'rb') as opened_file:
        state_dict = torch.load(opened_file, map_location="cpu")
    if 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict, strict=False)
    ## 
    # model_path = "/data/guohua/BeiJing/zzz/RobotPVR/Project/runnings/RnC/clipthenrnc/ckpt_0ep.pth"
    # print(f"===========Loading '{model_path}' Model==============")
    # with open(model_path, 'rb') as opened_file:
    #     state_dict = torch.load(opened_file, map_location="cpu")
    # if 'model' in state_dict:
    #     state_dict = state_dict['model']
    # model.load_state_dict(state_dict, strict=False)
    
    print("========= Load Successfully ========")
    return model.eval()
    
    
