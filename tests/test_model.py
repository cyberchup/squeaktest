"""Tests for the AntiDeepfake port (step 5).

Most tests use a tiny randomly initialized network and run anywhere, including CI. The parity
tests need the real weights (fetch them into SQUEAKTEST_DATA_DIR) and are skipped otherwise.
They compare against logits computed by NII's original fairseq code
(scripts/parity_reference.py), stored in tests/data/.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from signals import parity_signals

from squeaktest.config import Settings
from squeaktest.inference import analyze
from squeaktest.model import (
    MMS_300M,
    AntiDeepfakeNet,
    Detector,
    ModelIntegrityError,
    ModelNotFoundError,
    convert_state_dict,
    load_detector,
    normalize,
    verify_weights,
    weights_path,
)

TINY = dataclasses.replace(
    MMS_300M, name="tiny", hidden_size=32, num_layers=2, num_heads=2, intermediate_size=64
)
REFERENCE = Path(__file__).parent / "data" / "antideepfake_mms300m_reference.json"


def fairseq_keys(num_layers: int) -> list[str]:
    """Tensor names in NII's checkpoint layout, for a model with `num_layers` layers."""
    ssl = "m_ssl.model."
    keys = []
    for i in range(7):
        conv = f"{ssl}feature_extractor.conv_layers.{i}."
        keys += [conv + "0.weight", conv + "0.bias", conv + "2.1.weight", conv + "2.1.bias"]
    for name in ["layer_norm", "post_extract_proj", "encoder.layer_norm"]:
        keys += [f"{ssl}{name}.weight", f"{ssl}{name}.bias"]
    keys += [f"{ssl}encoder.pos_conv.0.{p}" for p in ("bias", "weight_g", "weight_v")]
    for i in range(num_layers):
        layer = f"{ssl}encoder.layers.{i}."
        for part in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "self_attn_layer_norm",
            "fc1",
            "fc2",
            "final_layer_norm",
        ]:
            keys += [layer + part + ".weight", layer + part + ".bias"]
    keys += [f"{ssl}{n}" for n in ("quantizer.vars", "quantizer.weight_proj.weight")]
    keys += [f"{ssl}{n}" for n in ("project_q.weight", "final_proj.weight", "mask_emb")]
    keys += ["proj_fc.weight", "proj_fc.bias"]
    return keys


# --- Weight renaming ---------------------------------------------------------------------


def test_conversion_covers_every_parameter_of_the_network():
    converted = convert_state_dict({k: torch.zeros(1) for k in fairseq_keys(num_layers=2)})
    assert set(converted) == set(AntiDeepfakeNet(TINY).state_dict())


def test_pretraining_only_tensors_are_dropped():
    converted = convert_state_dict({k: torch.zeros(1) for k in fairseq_keys(num_layers=2)})
    assert not any("quantizer" in k or "project_q" in k or "mask" in k for k in converted)


@pytest.mark.parametrize(
    ("fairseq", "ours"),
    [
        (
            "m_ssl.model.encoder.layers.3.self_attn.k_proj.weight",
            "wav2vec2.encoder.layers.3.attention.k_proj.weight",
        ),
        (
            "m_ssl.model.encoder.layers.0.fc1.bias",
            "wav2vec2.encoder.layers.0.feed_forward.intermediate_dense.bias",
        ),
        (
            "m_ssl.model.feature_extractor.conv_layers.6.2.1.weight",
            "wav2vec2.feature_extractor.conv_layers.6.layer_norm.weight",
        ),
        ("m_ssl.model.layer_norm.weight", "wav2vec2.feature_projection.layer_norm.weight"),
        (
            "m_ssl.model.encoder.pos_conv.0.weight_g",
            "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0",
        ),
        ("proj_fc.weight", "classifier.weight"),
    ],
)
def test_individual_renames(fairseq: str, ours: str):
    assert ours in convert_state_dict({fairseq: torch.zeros(1)})


@pytest.mark.parametrize("key", ["m_ssl.model.something_new.weight", "unexpected.weight"])
def test_unknown_tensors_are_an_error(key: str):
    with pytest.raises(ValueError, match="unexpected tensors"):
        convert_state_dict({key: torch.zeros(1)})


# --- Preprocessing and batching ----------------------------------------------------------


def test_normalize_gives_zero_mean_unit_variance_per_waveform():
    batch = torch.stack([torch.linspace(-3, 9, 1000), torch.full((1000,), 0.1) + torch.randn(1000)])
    out = normalize(batch)
    assert torch.allclose(out.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert torch.allclose(out.var(dim=1, unbiased=False), torch.ones(2), atol=1e-3)


@pytest.fixture(scope="module")
def tiny_detector() -> Detector:
    torch.manual_seed(0)
    return Detector(TINY, AntiDeepfakeNet(TINY).eval(), torch.device("cpu"))


def test_mixed_lengths_keep_their_order(tiny_detector: Detector):
    sig = parity_signals()
    windows = [sig["tone_440hz_4s"], sig["tone_880hz_1s"], sig["chirp_100_4000hz_4s"]]
    together = tiny_detector.logits(windows)
    alone = np.vstack([tiny_detector.logits([w]) for w in windows])
    np.testing.assert_allclose(together, alone, atol=1e-5)


def test_scores_are_probabilities(tiny_detector: Detector):
    scores = tiny_detector([parity_signals()["lcg_noise_2s"]] * 3)
    assert len(scores) == 3
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores)


# --- Weights on disk ---------------------------------------------------------------------


def test_missing_weights_explain_what_to_do(tmp_path: Path):
    with pytest.raises(ModelNotFoundError, match="Download them first"):
        load_detector(data_dir=tmp_path)


def test_weights_must_match_the_published_checksum(tmp_path: Path):
    content = b"pretend these are weights"
    spec = dataclasses.replace(
        TINY, sha256=hashlib.sha256(content).hexdigest(), size_bytes=len(content)
    )
    path = weights_path(spec, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    verify_weights(spec, path)  # matches: no error

    path.write_bytes(content[:-1] + b"!")  # same size, different bytes
    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        verify_weights(spec, path)

    path.write_bytes(content + b"extra")
    with pytest.raises(ModelIntegrityError, match="size"):
        verify_weights(spec, path)


# --- Parity with NII's original code (needs the real weights) ----------------------------

DATA_DIR = Settings.from_env().data_dir
needs_weights = pytest.mark.skipif(
    not weights_path(MMS_300M, DATA_DIR).exists(),
    reason="MMS-300M weights not downloaded (set SQUEAKTEST_DATA_DIR)",
)


def _reference_logits() -> dict[str, np.ndarray]:
    return {k: np.array(v) for k, v in json.loads(REFERENCE.read_text())["logits"].items()}


@needs_weights
@pytest.mark.parametrize(
    ("device", "tolerance"),
    [
        ("cpu", 2e-4),
        pytest.param(
            "cuda",
            5e-4,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
        ),
    ],
)
def test_port_matches_niis_fairseq_model(device: str, tolerance: float):
    detector = load_detector(data_dir=DATA_DIR, device=device)
    reference = _reference_logits()
    signals = parity_signals()
    names = sorted(reference)
    ours = detector.logits([signals[n] for n in names])
    for i, name in enumerate(names):
        np.testing.assert_allclose(ours[i], reference[name], atol=tolerance, err_msg=name)


@needs_weights
def test_real_detector_plugs_into_windowed_analysis():
    detector = load_detector(data_dir=DATA_DIR, device="cpu")
    audio = np.concatenate([parity_signals()["buzz_150hz_am_4_5s"]] * 2)
    result = analyze(audio, 16_000, detector)
    assert result.score is not None and 0.0 <= result.score <= 1.0
    assert all(w.score is not None for w in result.windows)


def test_batches_are_capped_and_results_unchanged(tiny_detector: Detector):
    windows = [parity_signals()["tone_440hz_4s"] * (1 + i / 10) for i in range(5)]
    sizes: list[int] = []
    original_forward = tiny_detector.net.forward

    def recording_forward(waveforms):
        sizes.append(waveforms.shape[0])
        return original_forward(waveforms)

    tiny_detector.net.forward = recording_forward
    try:
        tiny_detector.max_batch = 2
        capped = tiny_detector.logits(windows)
        tiny_detector.max_batch = 8
        uncapped = tiny_detector.logits(windows)
    finally:
        del tiny_detector.net.forward
    assert sizes[:3] == [2, 2, 1]  # 5 windows in batches of at most 2
    np.testing.assert_allclose(capped, uncapped, atol=1e-5)
