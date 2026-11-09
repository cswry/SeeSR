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
