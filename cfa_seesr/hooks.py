"""Attach the method to a loaded upstream pipeline without editing upstream source.

* Soft-prompt strength (Component A): a forward hook on every `image_attentions` block of the UNet and the
  ControlNet. Each block is a diffusers Transformer2DModel whose output already contains its input as a
  residual (output = input + Delta), so the hook returns input + s * (output - input). s = 1 returns the
  original output object untouched.
* Spatial semantic reliability (SSR): the same hook with s replaced by a map s(x) of shape [1, 1, H, W]
  (broadcast over channels and over the CFG batch), when MethodState.spatial is set.
* LR schedule (Component B): wraps `controlnet.forward` and replaces `conditioning_scale` by
  conditioning_scale * f(t), reading the actual timestep argument. Nothing else in the call changes.
* SSR tile tracking: wraps `pipe.prepare_image` (records the full conditioning image and the latent size)
  and `controlnet.forward` (identifies the tile of each call, see install_spatial_tracking).
* SFR-OC (V2, observation_consistency.py): wraps `pipe.prepare_image` (builds the reliability maps at
  latent resolution; heuristic and/or LORNet, loaded from a validated checkpoint at installation), `pipe.vae.encode` (captures the pipeline's own LR posterior for
  `reference: pipeline`; the default `extra_encode` encodes the conditioning image once more) and
  `pipe.scheduler.step` (runs the unchanged step on the unchanged model output and adds c0(t) * Delta to its
  result; see observation_consistency.py). No UNet or ControlNet call is added.

Everything is installed only for an enabled, non-identity component and removed with `MethodHooks.remove()`
(wrappers are unwound in reverse order, so they compose).
"""
import functools

from .lr_guidance import schedule_factor


class MethodState:
    """Per-image values read by the hooks; written by the prompt step and the runner."""

    def __init__(self):
        self.soft_scale = 1.0
        self.spatial = None       # SpatialMap or None (scalar path)
        self.spatial_result = None
        self.semantic_reliability = None   # component A's image-level C for the current image (or None)
        self.sfroc = None         # SFROCController (V2) or None
        self.image_key = ""
        self.lr_image = None      # original LR image of the current image (SFR-OC noise estimate) or None
        self.lr_log = []          # [(timestep, factor)] for the current pipeline call
        self.reset_call()

    def reset_call(self):
        self.lr_log = []
        self.full_image = None    # conditioning image of the current pipeline call [B, 3, H, W]
        self.tile_log = []        # [(timestep, region or None)] per ControlNet call
        self._tile_t = None
        self._tile_count = 0


def image_attention_blocks(model):
    """All image cross-attention transformer blocks of a SeeSR UNet / ControlNet."""
    blocks = []
    for name, module in model.named_modules():
        if name.endswith("image_attentions") and hasattr(module, "__iter__"):
            blocks.extend(module)
    return blocks


class MethodHooks:
    def __init__(self):
        self._handles = []
        self._patched = []        # [(obj, attr, had_instance_attr, previous)] in installation order

    def _patch(self, obj, attr, new):
        had = attr in vars(obj)
        self._patched.append((obj, attr, had, vars(obj).get(attr)))
        setattr(obj, attr, new)   # instance attribute shadows the class method

    def install_soft_prompt_scaling(self, pipe, state):
        blocks = image_attention_blocks(pipe.unet) + image_attention_blocks(pipe.controlnet)
        if not blocks:
            raise RuntimeError("no image_attentions blocks found: SeeSR soft-prompt pathway missing "
                               "(use_image_cross_attention must be true)")

        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs["hidden_states"]
            out = output[0] if isinstance(output, tuple) else output.sample
            spatial = state.spatial
            if spatial is not None:
                if out.ndim != 4:
                    raise RuntimeError(f"image-attention output has shape {tuple(out.shape)}; expected [B, C, H, W]")
                s = spatial.block_scale(out.shape[-2], out.shape[-1], out.device, out.dtype)
            else:
                s = state.soft_scale
                if s == 1.0:
                    return output
            new = x + s * (out - x)
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            output.sample = new
            return output

        for b in blocks:
            self._handles.append(b.register_forward_hook(hook, with_kwargs=True))
        return len(blocks)

    def install_lr_schedule(self, pipe, cfg, state):
        controlnet = pipe.controlnet
        original = controlnet.forward
        max_t = pipe.scheduler.config.num_train_timesteps - 1

        @functools.wraps(original)
        def forward(sample, timestep, *args, conditioning_scale=1.0, **kwargs):
            factor = schedule_factor(timestep, cfg, max_t)
            state.lr_log.append((float(timestep), factor))
            return original(sample, timestep, *args, conditioning_scale=conditioning_scale * factor, **kwargs)

        self._patch(controlnet, "forward", forward)

    def install_spatial_tracking(self, pipe, state, tile_overlap):
        """Give the SSR map the latent size and, for tiled sampling, the tile of every ControlNet call.

        The pipeline calls ControlNet and then the UNet once per tile, in a fixed order, and passes neither
        coordinates; the ControlNet receives the pixel tile of the conditioning image. The tile is taken
        from the re-implemented enumeration (spatial_reliability.tile_regions) by call order within a
        timestep, and VERIFIED by comparing the received pixel tile with the same slice of the full
        conditioning image; any mismatch raises instead of applying a wrong map. The region is kept for the
        following UNet call.
        """
        import torch

        from .spatial_reliability import resolve_mean_matched, tile_regions

        original_prepare = pipe.prepare_image

        @functools.wraps(original_prepare)
        def prepare_image(*a, **k):
            image = original_prepare(*a, **k)
            state.full_image = image
            h, w = image.shape[-2] // 8, image.shape[-1] // 8
            res = state.spatial_result
            if res is not None and res.mode == "mean_matched":
                state.soft_scale = resolve_mean_matched(res, h, w)
            if state.spatial is not None:
                state.spatial.set_latent_size(h, w)
                state.spatial.set_region(None)
            return image

        self._patch(pipe, "prepare_image", prepare_image)

        controlnet = pipe.controlnet
        original_forward = controlnet.forward

        @functools.wraps(original_forward)
        def forward(sample, timestep, *args, **kwargs):
            if state.spatial is not None:
                cond = kwargs.get("controlnet_cond")
                full = state.full_image
                if cond is None or full is None:
                    raise RuntimeError("SSR tile tracking: conditioning image not available")
                if tuple(cond.shape[-2:]) == tuple(full.shape[-2:]):
                    region = None
                else:
                    t = float(timestep)
                    if state._tile_t != t:
                        state._tile_t, state._tile_count = t, 0
                    h, w = full.shape[-2] // 8, full.shape[-1] // 8
                    regions = tile_regions(h, w, cond.shape[-1] // 8, tile_overlap)
                    if state._tile_count >= len(regions):
                        raise RuntimeError("SSR tile tracking: more ControlNet calls than tiles in one step")
                    region = regions[state._tile_count]
                    state._tile_count += 1
                    y0, y1, x0, x1 = region
                    if not torch.equal(cond, full[..., y0 * 8:y1 * 8, x0 * 8:x1 * 8]):
                        raise RuntimeError(f"SSR tile tracking: tile {region} does not match the pixels the "
                                           "pipeline passed; refusing to apply a misaligned map")
                state.spatial.set_region(region)
                state.tile_log.append((float(timestep), region))
            return original_forward(sample, timestep, *args, **kwargs)

        self._patch(controlnet, "forward", forward)

    def install_observation_consistency(self, pipe, cfg, state):
        """SFR-OC (V2). Raises ValueError for an unsupported scheduler configuration (before any run).

        Per denoising step no tensor is converted to a Python value (see observation_consistency.py,
        "Synchronisation"); checks are deferred to SFROCController.summary().
        """
        import torch

        from .observation_consistency import (SFROCController, apply_to_step_output, apply_x0_delta,
                                              check_scheduler, correction_delta, gamma_at, step_diagnostics,
                                              x0_coefficient)

        prediction_type = check_scheduler(pipe.scheduler)
        max_t = pipe.scheduler.config.num_train_timesteps - 1
        lornet, lornet_meta = None, None
        if cfg.uses_learned:                               # validated checkpoint or an exception: never random
            from .lornet import checkpoint_sha256, load_lornet
            device = next(pipe.unet.parameters()).device
            lornet, lornet_meta = load_lornet(cfg.learned.checkpoint, device)
            lornet_meta = {**lornet_meta, "path": str(cfg.learned.checkpoint),
                           "sha256": checkpoint_sha256(cfg.learned.checkpoint)}
        ctrl = SFROCController(cfg, max_t, prediction_type, lornet, lornet_meta)
        state.sfroc = ctrl
        scheduler_name = type(pipe.scheduler).__name__

        original_encode = pipe.vae.encode

        @functools.wraps(original_encode)
        def encode(*a, **k):
            out = original_encode(*a, **k)
            if ctrl.encode_dist is None:
                ctrl.encode_dist = out.latent_dist if hasattr(out, "latent_dist") else out[0]
            return out

        self._patch(pipe.vae, "encode", encode)

        original_prepare = pipe.prepare_image

        @functools.wraps(original_prepare)
        def prepare_image(*a, **k):
            image = original_prepare(*a, **k)
            ctrl.reset()
            ctrl.image = image
            ctrl.lr_image = getattr(state, "lr_image", None)
            h, w = image.shape[-2] // 8, image.shape[-1] // 8
            ctrl.analysis_timer.start(image.device)
            if state.spatial is not None:                      # SSR map at latent resolution
                state.spatial.set_latent_size(h, w)
                S = state.spatial.latent.float()
                source = "ssr_map"
            elif state.semantic_reliability is not None:       # component A's image-level reliability
                S = torch.full((h, w), float(state.semantic_reliability), device=image.device)
                source = "semantic_confidence"
            else:
                S = torch.full((h, w), cfg.fusion.semantic_default, device=image.device)
                source = "default"
            ctrl.build_maps(image, (h, w), S.to(image.device), key=state.image_key)
            ctrl.analysis_timer.stop()
            ctrl.map_info["semantic_source"] = source
            return image

        self._patch(pipe, "prepare_image", prepare_image)

        scheduler = pipe.scheduler
        original_step = scheduler.step

        def reference_latent(sample):
            if ctrl.z_ref is None:
                vae = pipe.vae
                if cfg.reference == "pipeline":                # opt-in: reuse the pipeline's own LR encode
                    dist = ctrl.encode_dist
                    if dist is None:
                        raise RuntimeError("SFR-OC: reference 'pipeline' needs the pipeline's LR-start encode "
                                           "(start_point 'lr'); use reference 'extra_encode'")
                    ctrl.reference_source = "pipeline_encode"
                    mean = dist.mean
                    if mean.shape[0] > 1:                      # CFG copies of the same image: deferred check
                        ctrl.checks["reference_cfg_identical"] = (mean == mean[0:1]).all()
                else:                                          # default: encode conditioning item 0 once
                    # (its CFG copies are checked identical when the maps are built: cfg_batch_identical)
                    ctrl.encode_timer.start(sample.device)
                    with torch.no_grad():
                        x = (ctrl.image[0:1] * 2 - 1).to(device=vae.device, dtype=vae.dtype)
                        mean = original_encode(x).latent_dist.mean
                    ctrl.encode_timer.stop()
                    ctrl.reference_source = "extra_encode"
                ctrl.z_ref = (mean[0:1].float() * vae.config.scaling_factor).detach()
            if tuple(ctrl.z_ref.shape[-2:]) != tuple(sample.shape[-2:]):
                raise RuntimeError(f"SFR-OC: reference latent {tuple(ctrl.z_ref.shape)} does not match the "
                                   f"sample {tuple(sample.shape)}")
            return ctrl.z_ref

        def ddim_eta(a, k):
            if scheduler_name != "DDIMScheduler":
                return 0.0
            if "eta" in k:
                return float(k["eta"])
            return float(a[0]) if a else 0.0

        @functools.wraps(original_step)
        def step(model_output, timestep, sample, *a, **k):
            t = ctrl.next_timestep(scheduler, timestep)        # host value, no device sync
            gamma = gamma_at(t, cfg, max_t)
            if gamma == 0.0:                                   # exact no-op: original call, original arguments
                ctrl.carry = None
                ctrl.step_log.append({"t": t, "gamma": 0.0})
                return original_step(model_output, timestep, sample, *a, **k)
            if ctrl.W is None or tuple(ctrl.W.shape[-2:]) != tuple(sample.shape[-2:]):
                raise RuntimeError("SFR-OC: reliability map missing or of the wrong size for this step")
            a_t = float(scheduler.alphas_cumprod[t])           # host tensor on CPU in diffusers: no device sync
            if not 0.0 < a_t < 1.0:
                raise ValueError(f"SFR-OC: alphas_cumprod[{t}] = {a_t} is outside (0, 1)")
            z_ref = reference_latent(sample)                  # timed separately (encode_ms), not as correction
            ctrl.correction_timer.start(sample.device)
            delta, x0, r = correction_delta(model_output, sample, a_t, z_ref, ctrl.W, gamma, cfg, prediction_type)
            diag = step_diagnostics(t, gamma, r, delta)
            if cfg.precision in ("output", "output_carry"):
                c0 = x0_coefficient(scheduler, t, ddim_eta(a, k))
                diag["c0"] = c0
                ctrl.correction_timer.stop()
                out = original_step(model_output, timestep, sample, *a, **k)
                ctrl.correction_timer.start(sample.device)
                carry = ctrl.carry if cfg.precision == "output_carry" else None
                out, rem = apply_to_step_output(out, delta, c0, x0, carry=carry, return_carry=True)
                ctrl.carry = rem if cfg.precision == "output_carry" else None
                ctrl.correction_timer.stop()
            else:
                corrected = apply_x0_delta(model_output.float(), delta, a_t, prediction_type).to(model_output.dtype)
                ctrl.correction_timer.stop()
                out = original_step(corrected, timestep, sample, *a, **k)
            ctrl.step_log.append(diag)
            return out

        self._patch(scheduler, "step", step)
        return ctrl

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for obj, attr, had, prev in reversed(self._patched):
            if had:
                setattr(obj, attr, prev)
            else:
                delattr(obj, attr)
        self._patched = []
