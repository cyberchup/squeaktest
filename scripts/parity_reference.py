"""Compute reference logits with NII's original fairseq model definition.

Run this in a separate environment that has NII's stack (Python 3.9, fairseq 0.12.2,
safetensors). It mirrors the inference code on NII's model card for MMS-300M-AntiDeepfake,
and loads the weights with strict key matching. squeaktest's transformers port must reproduce
its output (tests/test_model.py).

    python parity_reference.py WEIGHTS.safetensors INPUTS.npz OUTPUT.json

INPUTS.npz holds 1-D float32 waveforms at 16 kHz (from tests/signals.py).
"""

import json
import platform
import sys

import fairseq
import numpy as np
import torch
from fairseq.models.wav2vec import Wav2Vec2Config, Wav2Vec2Model
from safetensors.torch import load_file


class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0,
        )
        self.model = Wav2Vec2Model(cfg)


class DeepfakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_ssl = SSLModel()
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(1024, 2)

    def forward(self, wav):
        emb = self.m_ssl.model(wav, mask=False, features_only=True)["x"]  # [B, T, D]
        pooled = self.adap_pool1d(emb.transpose(1, 2)).squeeze(-1)  # [B, D]
        return self.proj_fc(pooled)  # [B, 2]: (fake, real)


def main(weights, inputs, output):
    net = DeepfakeDetector()
    net.load_state_dict(load_file(weights), strict=True)
    net.eval()
    data = np.load(inputs)
    logits = {}
    with torch.no_grad():
        for name in sorted(data.files):
            wav = torch.from_numpy(data[name]).float()
            wav = torch.nn.functional.layer_norm(wav, wav.shape)  # NII's preprocessing
            logits[name] = net(wav.unsqueeze(0))[0].tolist()
    result = {
        "logits": logits,
        "reference": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "fairseq": fairseq.__version__,
            "numpy": np.__version__,
        },
    }
    with open(output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:4])
