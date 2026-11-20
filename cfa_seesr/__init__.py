"""Confidence- and Fidelity-Aware SeeSR (training-free inference extension).

Component A  semantic_confidence.py  confidence-aware DAPE tag selection and soft-prompt strength
Component B  lr_guidance.py          timestep-dependent ControlNet (LR) conditioning strength
SSR          spatial_reliability.py  spatial semantic reliability: component A's strength distributed over
                                     the image using DAPE's tagging-head attention
V2 SFR-OC    observation_reliability.py, frequency.py, observation_consistency.py
                                     reliability-guided, frequency-selective observation consistency in the
                                     scheduler step (no extra UNet/ControlNet pass)
V2 LORNet    lornet.py, latent_targets.py   learned observation-reliability map (trained offline in
                                     experiments/training/ with the SeeSR backbone frozen)
Attachment   hooks.py                forward hooks / wrappers on a loaded upstream pipeline

Nothing in the upstream SeeSR code is modified. With every component disabled (the default) no hook is
installed and the upstream functions run unchanged. See README.md in this folder for the formulation.
"""
from dataclasses import asdict, dataclass, field

from .lr_guidance import LRGuidanceConfig, mean_factor, schedule_factor
from .semantic_confidence import SemanticConfidenceConfig, SemanticResult, extract_semantics, select_tags
from .spatial_reliability import SpatialSemanticConfig, build_spatial, capture_dape
from .observation_consistency import SFROCConfig

__all__ = ["MethodConfig", "SemanticConfidenceConfig", "LRGuidanceConfig", "SemanticResult",
           "extract_semantics", "select_tags", "schedule_factor", "mean_factor", "semantic_validation_prompt",
           "SpatialSemanticConfig", "spatial_validation_prompt", "SFROCConfig", "METHOD_VERSION",
           "method_signature"]

SECTIONS = ("semantic_confidence", "lr_guidance", "spatial_semantic", "sfr_oc")

# Version of this package's formulation, recorded in run_metadata.json. 2.1: audit remediation of SFR-OC
# (two-pass band-limited Delta, output-space application, noise-compensated observation reliability).
# 2.1.1: GPU runtime fixes from the V2.1 Colab pilots (device/dtype in SSR, fp16-safe quantile, CPU copy of the
# SSR grid before saving) and sfr_oc.reference default 'extra_encode' ('auto' removed).
METHOD_VERSION = "2.1.1"


def method_signature(method):
    """'upstream' (nothing active), 'cfa-seesr-v1' (A / B / SSR only) or 'cfa-seesr-v2' (SFR-OC active)."""
    if method.sfroc_active:
        return "cfa-seesr-v2"
    if method.is_baseline:
        return "upstream"
    return "cfa-seesr-v1"


@dataclass(frozen=True)
class MethodConfig:
    semantic_confidence: SemanticConfidenceConfig = field(default_factory=SemanticConfidenceConfig)
    lr_guidance: LRGuidanceConfig = field(default_factory=LRGuidanceConfig)
    spatial_semantic: SpatialSemanticConfig = field(default_factory=SpatialSemanticConfig)
    sfr_oc: SFROCConfig = field(default_factory=SFROCConfig)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        unknown = set(d) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown method sections: {sorted(unknown)}")
        return cls(SemanticConfidenceConfig.from_dict(d.get("semantic_confidence")),
                   LRGuidanceConfig.from_dict(d.get("lr_guidance")),
                   SpatialSemanticConfig.from_dict(d.get("spatial_semantic")),
                   SFROCConfig.from_dict(d.get("sfr_oc")))

    @property
    def semantic_active(self):
        return self.semantic_confidence.enabled

    @property
    def lr_active(self):
        return self.lr_guidance.enabled and not self.lr_guidance.is_identity()

    @property
    def spatial_active(self):
        return self.spatial_semantic.enabled

    @property
    def soft_scale_range(self):
        """(s_min, s_max) of the soft-prompt hook: component A's when it is on, otherwise SSR's."""
        if self.semantic_active:
            return self.semantic_confidence.soft_scale_min, self.semantic_confidence.soft_scale_max
        if self.spatial_active:
            return self.spatial_semantic.soft_scale_min, self.spatial_semantic.soft_scale_max
        return 1.0, 1.0

    @property
    def sfroc_active(self):
        """V2 SFR-OC hooks are installed (also with strength 0: the no-op control runs the hooks)."""
        return self.sfr_oc.enabled

    @property
    def is_baseline(self):
        return not (self.semantic_active or self.lr_active or self.spatial_active or self.sfroc_active)

    def to_dict(self):
        return asdict(self)


def semantic_validation_prompt(upstream, args, image, model, cfg):
    """Drop-in for upstream test_seesr.get_validation_prompt with the confidence rule applied.

    Same preprocessing (upstream.tensor_transforms + upstream.ram_transforms) and the same prompt format
    f"{tags}, {args.prompt},". Returns (prompt, image_embeds, SemanticResult).
    """
    device = next(model.parameters()).device
    lq = upstream.ram_transforms(upstream.tensor_transforms(image).unsqueeze(0).to(device))
    embeds, result = extract_semantics(model, lq, cfg)
    return f"{result.prompt_tags}, {args.prompt},", embeds, result


def spatial_validation_prompt(upstream, args, image, model, sem_cfg, ssr_cfg, key=""):
    """semantic_validation_prompt plus the SSR map, from the SAME single DAPE pass.

    The tagging-head attention and the logits are read by passive hooks during extract_semantics; the
    prompt, the image embeddings and component A's result are exactly those semantic_validation_prompt
    returns. Returns (prompt, image_embeds, SemanticResult, SpatialResult).
    """
    device = next(model.parameters()).device
    lq = upstream.ram_transforms(upstream.tensor_transforms(image).unsqueeze(0).to(device))
    with capture_dape(model, ssr_cfg.attention_layer) as cap:
        embeds, result = extract_semantics(model, lq, sem_cfg)
    if "attention" not in cap or "logits" not in cap:
        raise RuntimeError("DAPE pass did not reach the tagging-head attention / fc layer")
    if cap["attention"].shape[-1] != embeds.shape[1]:
        raise RuntimeError(f"attention keys ({cap['attention'].shape[-1]}) != image tokens ({embeds.shape[1]})")
    spatial = build_spatial(cap["attention"], cap["logits"], model, sem_cfg, ssr_cfg, result, key)
    cap.clear()   # drop the [1, heads, K, T] tensor now
    return f"{result.prompt_tags}, {args.prompt},", embeds, result, spatial
