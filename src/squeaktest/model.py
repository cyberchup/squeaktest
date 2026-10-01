"""Load NII's AntiDeepfake detector into Hugging Face `transformers`, without fairseq.

NII publishes the weights in fairseq's layout and loads them with fairseq, an unmaintained
library that doesn't install on Python 3.12. The network itself is a standard wav2vec 2.0
model, which `transformers` implements, so this module rebuilds the same network there and
renames the weights to match. The port is checked against NII's own code: see
tests/test_model.py and docs/decisions.md (D8).

The detector, as NII defines it:
1. Normalize the waveform to zero mean and unit variance (layer norm over the whole input).
2. Run wav2vec 2.0 and take its final layer's output: one 1024-dim vector per 20 ms.
3. Average those vectors over time, then apply Linear(1024, 2).
4. Softmax: index 0 is "fake", index 1 is "real". squeaktest's score is P(fake).

Weights are downloaded only when asked (fetch_weights), pinned to a commit, and checked
against a published SHA-256 before every load.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Model

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ModelSpec:
    """Where a detector's weights come from, under what license, and its architecture."""

    name: str
    repo_id: str
    revision: str  # a commit hash, so the weights can't change underneath us
    filename: str
    sha256: str
    size_bytes: int
    license: str
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    # Large models normalize inside each block ("layer norm first") and after every
    # convolution; base models normalize after each block and only after the first
    # convolution (group norm), with no convolution biases.
    norm_first: bool = True
    conv_norm: str = "layer"  # "layer" or "group"
    conv_bias: bool = True

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/tree/{self.revision}"


MMS_300M = ModelSpec(
    name="antideepfake-mms-300m",
    repo_id="nii-yamagishilab/mms-300m-anti-deepfake",
    revision="7928e119e09f616b2ce698892e00ca22dbaa40bf",
    filename="model.safetensors",
    sha256="9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224",
    size_bytes=1_269_622_944,
    license="CC BY-NC-SA 4.0",
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    intermediate_size=4096,
)

W2V_LARGE = ModelSpec(
    name="antideepfake-w2v-large",
    repo_id="nii-yamagishilab/wav2vec-large-anti-deepfake",
    revision="7ccfcbb27d3098dafeb67596d4bb26e51e4627fc",
    filename="model.safetensors",
    sha256="b27943fefaff677bc95890051cf27b14fd72ab66e55eca6d3395cdb5788c2bb5",
    size_bytes=1_269_622_944,
    license="CC BY-NC-SA 4.0",
    hidden_size=1024,
    num_layers=24,
    num_heads=16,
    intermediate_size=4096,
)

W2V_SMALL = ModelSpec(
    name="antideepfake-w2v-small",
    repo_id="nii-yamagishilab/wav2vec-small-anti-deepfake",
    revision="9a13264b5dcc827a8d5a4f8e01fccefa392f886b",
    filename="model.safetensors",
    sha256="828ee456122f86d5d631cb7895a10e5c62c78a4fcb8a8b1c42cb5838a9abcfe0",
    size_bytes=380_210_632,
    license="CC BY-NC-SA 4.0",
    hidden_size=768,
    num_layers=12,
    num_heads=12,
    intermediate_size=3072,
    norm_first=False,
    conv_norm="group",
    conv_bias=False,
)

MODELS: dict[str, ModelSpec] = {m.name: m for m in (MMS_300M, W2V_LARGE, W2V_SMALL)}
DEFAULT_MODEL = MMS_300M.name


class ModelNotFoundError(RuntimeError):
    """The weights haven't been downloaded yet."""


class ModelIntegrityError(RuntimeError):
    """The weights on disk don't match the published checksum."""


def weights_path(spec: ModelSpec, data_dir: Path) -> Path:
    return Path(data_dir) / "models" / spec.name / spec.filename


def fetch_weights(spec: ModelSpec, data_dir: Path) -> Path:
    """Download the pinned weights into data_dir (if needed) and verify them."""
    path = weights_path(spec, data_dir)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=path.parent,
        )
    verify_weights(spec, path)
    return path


def verify_weights(spec: ModelSpec, path: Path) -> None:
    """Check the file's size and SHA-256 against the values published with the pinned commit."""
    if path.stat().st_size != spec.size_bytes:
        raise ModelIntegrityError(f"{spec.name}: weights file has the wrong size")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    _check_digest(spec, digest.hexdigest())


def _verified_bytes(spec: ModelSpec, path: Path) -> bytes:
    """Read the weights once and verify those exact bytes, so what's loaded is what's checked."""
    data = path.read_bytes()
    if len(data) != spec.size_bytes:
        raise ModelIntegrityError(f"{spec.name}: weights file has the wrong size")
    _check_digest(spec, hashlib.sha256(data).hexdigest())
    return data


def _check_digest(spec: ModelSpec, hexdigest: str) -> None:
    if hexdigest != spec.sha256:
        raise ModelIntegrityError(f"{spec.name}: weights file does not match its SHA-256")


def wav2vec2_config(spec: ModelSpec) -> Wav2Vec2Config:
    """The transformers config equivalent to NII's fairseq configuration."""
    return Wav2Vec2Config(
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_layers,
        num_attention_heads=spec.num_heads,
        intermediate_size=spec.intermediate_size,
        # fairseq's extractor_mode, layer_norm_first and conv_bias
        feat_extract_norm=spec.conv_norm,
        do_stable_layer_norm=spec.norm_first,
        conv_bias=spec.conv_bias,
        layer_norm_eps=1e-5,
        # Inference only: no dropout, no layer drop, no masking.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=False,
        mask_time_prob=0.0,
    )


class AntiDeepfakeNet(nn.Module):
    """wav2vec 2.0 front-end, mean pooling over time, and a linear classifier."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(wav2vec2_config(spec))
        self.classifier = nn.Linear(spec.hidden_size, 2)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Map normalized waveforms [batch, samples] to logits [batch, 2] (fake, real)."""
        hidden = self.wav2vec2(waveforms).last_hidden_state  # [batch, frames, hidden]
        return self.classifier(hidden.mean(dim=1))


# fairseq name -> transformers name. The first matching pattern wins.
_RENAMES: list[tuple[str, str]] = [
    (
        r"feature_extractor\.conv_layers\.(\d+)\.0\.",
        r"wav2vec2.feature_extractor.conv_layers.\1.conv.",
    ),
    (  # large models: a layer norm after every convolution
        r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.",
        r"wav2vec2.feature_extractor.conv_layers.\1.layer_norm.",
    ),
    (  # base models: one group norm, after the first convolution only
        r"feature_extractor\.conv_layers\.0\.2\.",
        r"wav2vec2.feature_extractor.conv_layers.0.layer_norm.",
    ),
    (r"layer_norm\.", r"wav2vec2.feature_projection.layer_norm."),
    (r"post_extract_proj\.", r"wav2vec2.feature_projection.projection."),
    (r"encoder\.pos_conv\.0\.bias", r"wav2vec2.encoder.pos_conv_embed.conv.bias"),
    (
        r"encoder\.pos_conv\.0\.weight_g",
        r"wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    ),
    (
        r"encoder\.pos_conv\.0\.weight_v",
        r"wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ),
    (r"encoder\.layer_norm\.", r"wav2vec2.encoder.layer_norm."),
    (
        r"encoder\.layers\.(\d+)\.self_attn\.(q|k|v|out)_proj\.",
        r"wav2vec2.encoder.layers.\1.attention.\2_proj.",
    ),
    (r"encoder\.layers\.(\d+)\.self_attn_layer_norm\.", r"wav2vec2.encoder.layers.\1.layer_norm."),
    (
        r"encoder\.layers\.(\d+)\.fc1\.",
        r"wav2vec2.encoder.layers.\1.feed_forward.intermediate_dense.",
    ),
    (r"encoder\.layers\.(\d+)\.fc2\.", r"wav2vec2.encoder.layers.\1.feed_forward.output_dense."),
    (
        r"encoder\.layers\.(\d+)\.final_layer_norm\.",
        r"wav2vec2.encoder.layers.\1.final_layer_norm.",
    ),
]

# Pretraining-only parts of the fairseq checkpoint. Detection never uses them.
_UNUSED = re.compile(r"(quantizer\.|project_q\.|final_proj\.|mask_emb$)")

_SSL_PREFIX = "m_ssl.model."
_HEAD_PREFIX = "proj_fc."


def convert_state_dict(fairseq: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename NII's fairseq-layout weights to AntiDeepfakeNet's names.

    Every tensor must be either renamed or known to be unused; anything else is an error, so a
    checkpoint with an unexpected layout fails loudly instead of loading half-initialized.
    """
    converted: dict[str, torch.Tensor] = {}
    unknown: list[str] = []
    for key, tensor in fairseq.items():
        if key.startswith(_HEAD_PREFIX):
            converted["classifier." + key[len(_HEAD_PREFIX) :]] = tensor
            continue
        if not key.startswith(_SSL_PREFIX):
            unknown.append(key)
            continue
        name = key[len(_SSL_PREFIX) :]
        if _UNUSED.match(name):
            continue
        for pattern, replacement in _RENAMES:
            new_name, count = re.subn(rf"^{pattern}", replacement, name)
            if count:
                converted[new_name] = tensor
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"unexpected tensors in checkpoint: {sorted(unknown)[:5]}")
    return converted


def normalize(waveform: torch.Tensor) -> torch.Tensor:
    """Zero mean, unit variance per waveform, as NII's preprocessing does."""
    return nn.functional.layer_norm(waveform, waveform.shape[-1:])


class Detector:
    """A loaded detector. Call it with a batch of windows to get P(fake) for each.

    It satisfies inference.WindowScorer, so it can be passed straight to inference.analyze.
    """

    def __init__(
        self, spec: ModelSpec, net: AntiDeepfakeNet, device: torch.device, max_batch: int = 8
    ) -> None:
        self.spec = spec
        self.net = net
        self.device = device
        # Caps memory: the first convolution's output alone is about 26 MB per 4 s window, so
        # one batch for a 5-minute clip (149 windows) would need about 4 GB for that layer.
        self.max_batch = max_batch

    @torch.inference_mode()
    def logits(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """Raw (fake, real) logits for each window."""
        out = np.empty((len(windows), 2), dtype=np.float64)
        # Windows of equal length run together, up to max_batch at a time. Padding shorter ones
        # would change the time average the classifier sees, so different lengths run apart.
        by_length: dict[int, list[int]] = {}
        for i, w in enumerate(windows):
            by_length.setdefault(len(w), []).append(i)
        for same_length in by_length.values():
            for start in range(0, len(same_length), self.max_batch):
                indices = same_length[start : start + self.max_batch]
                batch = np.stack([windows[i] for i in indices]).astype(np.float32)
                logits = self.net(normalize(torch.from_numpy(batch).to(self.device)))
                out[indices] = logits.float().cpu().numpy()
        return out

    def __call__(self, windows: Sequence[np.ndarray]) -> list[float]:
        logits = torch.from_numpy(self.logits(windows))
        return torch.softmax(logits, dim=1)[:, 0].tolist()


def load_detector(
    name: str = DEFAULT_MODEL,
    data_dir: Path | None = None,
    device: str | None = None,
) -> Detector:
    """Load a downloaded, verified detector onto `device` (default: GPU if available)."""
    from squeaktest.config import Settings

    spec = MODELS[name]
    data_dir = data_dir if data_dir is not None else Settings.from_env().data_dir
    path = weights_path(spec, data_dir)
    if not path.exists():
        raise ModelNotFoundError(
            f"{spec.name} weights not found under {Path(data_dir) / 'models'}. "
            f"Download them first ({spec.size_bytes / 1e9:.2f} GB, license {spec.license})."
        )
    state = convert_state_dict(load(_verified_bytes(spec, path)))

    # Build on the "meta" device: shapes only, no memory and no random initialization (which
    # took about 20 s for 300M parameters), then assign the real weights in place.
    with torch.device("meta"):
        net = AntiDeepfakeNet(spec)
    net.load_state_dict(state, strict=True, assign=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if torch_device.type == "cuda":
        # cuDNN's default TF32 convolutions made GPU logits drift up to 2e-3 from NII's
        # reference; full float32 matches within 1e-4 at no measurable cost (docs/decisions.md D12).
        torch.backends.cudnn.allow_tf32 = False
    net.to(torch_device).eval()
    return Detector(spec, net, torch_device)
