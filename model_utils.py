# model_utils.py

import torch
import torch.nn as nn
import librosa
import numpy as np
import timm

from config import CFG


# ---------------- BirdCLEF EfficientNet Model ----------------
class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
        )

        # 获取特征维度
        if hasattr(self.backbone, "get_classifier"):
            in_features = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
        else:
            in_features = self.backbone.num_features
            self.backbone.classifier = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # (B,1,H,W) -> (B,3,H,W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


# ---------------- Load Model ----------------
def load_model(weights_path):
    """
    load_model("models/efficientnet_b2.pth")
    """
    # 读取 classes.txt
    labels = [l.strip() for l in open("classes.txt").readlines()]
    num_classes = len(labels)

    model = BirdCLEFModel(CFG.model_name, num_classes, pretrained=False)

    state = torch.load(weights_path, map_location=CFG.device)
    model.load_state_dict(state)
    model.to(CFG.device)
    model.eval()

    return model


# ---------------- Audio to Mel ----------------
def audio_to_mel(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=CFG.n_mels,
        fmin=CFG.fmin,
        fmax=CFG.fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


# ---------------- Predict Bird ----------------
def predict_bird(model, y, sr):
    """
    输入降噪后的 waveform，生成 MEL，并用模型预测
    返回：
        pred_label : 预测的名字
        prob       : 最大概率
        prob_dict  : 所有类别及概率
    """
    labels = [l.strip() for l in open("classes.txt").readlines()]

    # audio → mel
    mel = audio_to_mel(y, sr)
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(CFG.device)

    with torch.no_grad():
        logits = model(mel_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # (num_classes,)

    # Top-1
    top_idx = np.argmax(probs)
    pred_label = labels[top_idx]
    pred_prob = float(probs[top_idx])

    prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return pred_label, pred_prob, prob_dict
