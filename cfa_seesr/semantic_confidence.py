"""Component A: confidence-aware semantic conditioning for SeeSR's DAPE prompts.

What DAPE (ram/models/ram_lora.py, RAMLora) actually provides for an LR image x:
  * image embeddings  E = image_proj(visual_encoder(x))          shape [1, 1+N, 512]  (soft prompt)
  * tag logits        z_i, i = 1..K (K = 4585)                    from the tagging head
  * per-class thresholds tau_i (ram/data/ram_tag_list_threshold.txt), loaded in RAMLora.__init__
  * upstream selection  S0 = {i : sigmoid(z_i) > tau_i} \\ delete_tag_index  -> ", ".join(tags)

Confidence. The tagging head is a set of per-class logistic outputs p_i = sigmoid(z_i), and DAPE was
distilled to match RAM's probabilities on the HR image, so p_i is NOT a calibrated probability that the
object is present. What the model does define is a per-class operating point tau_i. We therefore measure
evidence relative to that operating point in the model's own score space (log-odds):

    m_i = z_i - logit(tau_i)                     log-odds margin over the class threshold
    c_i = sigmoid(m_i / T)                       normalised confidence, c_i > 1/2  <=>  p_i > tau_i

For T = 1, c_i = odds_i / (odds_i + odds_tau_i): the share of the class odds relative to the threshold odds.
T (temperature, log-odds units) sets how many nats of margin count as decisive. c_i is a monotone,
threshold-centred score, not a probability of correctness.

Tag selection. S(c_min) = {i in S0 : c_i > c_min}. c_min = 1/2 reproduces upstream exactly; c_min > 1/2
raises every class threshold by the same log-odds amount T*logit(c_min), which respects the class-specific
thresholds (a constant offset in probability space would not).

Conditioning strength. The aggregate reliability is computed over the upstream tag set S0 (independent of
c_min, so selection and strength are separate knobs):

    C = (aggregate_{i in S0} c_i - 1/2) / (1/2)  in [0, 1]     (0 if S0 is empty)
    s_sem = s_min + C (s_max - s_min)

s_sem scales the residual contribution of SeeSR's image-attention (soft-prompt) blocks; see hooks.py.
s_max defaults to 1 so the method never amplifies the pathway beyond its trained strength.

Fallback. If no tag survives selection the hard prompt is empty. SeeSR was trained with the tag prompt
dropped for 50% of samples (null_text_ratio=0.5 in train_seesr.py / dataloaders/paired_dataset.py), so an
empty prompt is an in-distribution input; if S0 is empty, C = 0 and the soft prompt uses s_min.
"""
from dataclasses import asdict, dataclass

AGGREGATES = ("mean", "min")


@dataclass(frozen=True)
class SemanticConfidenceConfig:
    enabled: bool = False
    temperature: float = 1.0          # T, log-odds units
    min_confidence: float = 0.5       # c_min; 0.5 == upstream tag selection
    soft_scale_min: float = 1.0       # s_min; with s_min == s_max == 1 the soft pathway is unchanged
    soft_scale_max: float = 1.0       # s_max
    aggregate: str = "mean"           # how c_i over S0 are combined: mean | min

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("semantic_confidence.temperature must be > 0")
        if not 0.0 < self.min_confidence < 1.0:
            raise ValueError("semantic_confidence.min_confidence must be in (0, 1)")
        if not 0.0 <= self.soft_scale_min <= self.soft_scale_max:
            raise ValueError("need 0 <= soft_scale_min <= soft_scale_max")
        if self.aggregate not in AGGREGATES:
            raise ValueError(f"semantic_confidence.aggregate must be one of {AGGREGATES}")

    @classmethod
    def from_dict(cls, d):
        return cls(**(d or {}))


def _logit(p):
    import torch
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.log(p) - torch.log1p(-p)


def tag_confidence(logits, thresholds, temperature=1.0):
    """c_i = sigmoid((z_i - logit(tau_i)) / T). logits, thresholds: tensors of shape [K] (or broadcastable)."""
    import torch
    margin = logits.float() - _logit(thresholds.float().to(logits.device))
    return torch.sigmoid(margin / temperature), margin


@dataclass
class SemanticResult:
    prompt_tags: str            # ", ".join(selected tags), same format as RAMLora.generate_tag
    upstream_tags: str          # tags upstream would have produced (S0)
    soft_scale: float           # s_sem
    reliability: float          # C in [0, 1]
    n_upstream: int
    n_selected: int
    tag_scores: list            # [(tag, p, tau, c)] for S0, in class-index order

    def summary(self):
        d = asdict(self)
        d.pop("tag_scores")
        return d


def select_tags(logits, thresholds, tag_list, delete_index, cfg):
    """Apply the confidence rule to one image's logits [K]. Returns SemanticResult (without embeddings)."""
    import torch
    probs = torch.sigmoid(logits.float())
    conf, _ = tag_confidence(logits, thresholds, cfg.temperature)
    passed = probs > thresholds.float().to(probs.device)          # exactly the upstream test
    if len(delete_index):
        passed[list(delete_index)] = False
    upstream = torch.nonzero(passed).flatten().tolist()            # class-index order, as np.argwhere upstream
    if cfg.min_confidence == 0.5:
        selected = upstream                                        # identical to upstream by construction
    else:
        selected = [i for i in upstream if conf[i].item() > cfg.min_confidence]
    scores = [(str(tag_list[i]), probs[i].item(), float(thresholds[i]), conf[i].item()) for i in upstream]
    reliability = aggregate_reliability([s[3] for s in scores], cfg.aggregate)
    return SemanticResult(
        prompt_tags=", ".join(str(tag_list[i]) for i in selected),
        upstream_tags=", ".join(str(tag_list[i]) for i in upstream),
        soft_scale=soft_prompt_scale(reliability, cfg),
        reliability=reliability, n_upstream=len(upstream), n_selected=len(selected), tag_scores=scores)


def aggregate_reliability(confidences, how="mean"):
    """C = (agg(c_i) - 1/2) / (1/2), clipped to [0, 1]; 0 for an empty set."""
    if not confidences:
        return 0.0
    agg = sum(confidences) / len(confidences) if how == "mean" else min(confidences)
    return min(1.0, max(0.0, (agg - 0.5) / 0.5))


def soft_prompt_scale(reliability, cfg):
    """s_sem = s_min + C (s_max - s_min), always within [s_min, s_max] (written as a convex combination
    so the endpoints are exact in floating point)."""
    c = min(1.0, max(0.0, float(reliability)))
    return (1.0 - c) * cfg.soft_scale_min + c * cfg.soft_scale_max


def extract_semantics(model, lq, cfg):
    """One DAPE pass (RAMLora.condition_forward) -> (image_embeds, SemanticResult).

    condition_forward(only_feature=False) runs the same visual encoder, projection and tagging head as
    upstream's two separate calls (generate_tag + generate_image_embeds), so both outputs are identical.
    """
    import torch
    with torch.no_grad():
        image_embeds, logits, _ = model.condition_forward(lq, only_feature=False)
    if logits.shape[0] != 1:
        raise ValueError("extract_semantics expects a single image")
    result = select_tags(logits[0], model.class_threshold, model.tag_list, model.delete_tag_index, cfg)
    return image_embeds, result
