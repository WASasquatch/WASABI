import torch
from typing import List, Tuple, Dict, Any, Optional

from nodes import CLIPTextEncode

from .utils import (
    safe_device_dtype_like,
    match_token_shapes,
    apply_keep_magnitude,
    scale_embeddings,
    maybe_mod_pooled,
    build_schedule,
    apply_scalar_or_schedule_mean,
    interp,
    apply_bimode_kernel,
    flatten_conditioning,
    get_reference_like,
    phase_to_reference,
    phase_blend_back,
)


class WASABI_ConditioningModulate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "Conditioning list from CLIP/T5 text encoders."
                }),
                "mode": ([
                    "bislerp", "bilerp", "scale",
                    "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp"
                ], {
                    "default": "bislerp", 
                    "tooltip": "scale: multiply.\nslerp/lerp: move toward reference.\nbislerp/bilerp: then blend back toward original."
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Global scale when no scale_schedule. >1 amplifies token magnitudes; <1 reduces."
                }),
                "blend_back": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Amount to blend results back toward ORIGINAL (0-1). Overridden per-token by blend_schedule."
                }),
                "ref_interp": (["slerp", "lerp"], {
                    "default": "slerp", 
                    "tooltip": "Interpolation toward reference embeddings."
                }),
                "blend_back_interp": (["slerp", "lerp"], {
                    "default": "slerp", 
                    "tooltip": "Interpolation for the blend-back phase."
                }),
                "keep_magnitude": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "Preserve original L2 norm after each interpolation phase."
                }),
                "eps": ("FLOAT", {
                    "default": 1e-8, 
                    "min": 1e-12, 
                    "max": 1e-3, 
                    "step": 1e-9,
                    "tooltip": "Numerical stability epsilon for normalization and SLERP."}),
            },
            "optional": {
                "schedule_options": ("DICT", {
                    "tooltip": "Optional schedules bundle from WASABI ScheduleOptions node."
                }),
                "reference_conditioning": ("CONDITIONING", {
                    "tooltip": "Target conditioning for slerp/lerp modes."
                }),
                "t_ref": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Global fraction toward reference when ref_schedule is empty."
                }),
                "advanced_options": ("DICT", {
                    "tooltip": "Optional advanced interpolation parameters from WASABI AdvancedOptions node."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "modulate"
    CATEGORY = "for_testing"
    DESCRIPTION = "Token-scheduled scaling/interpolation of CLIP/T5 conditioning with selectable interpolation and blend-back."

    def modulate(self,
                 conditioning,
                 mode: str,
                 scale: float,
                 blend_back: float,
                 keep_magnitude: bool,
                 eps: float,
                 reference_conditioning=None,
                 t_ref: float = 0.5,
                 ref_interp: str = "slerp",
                 blend_back_interp: str = "slerp",
                 ref_schedule: str = "",
                 scale_schedule: str = "",
                 blend_schedule: str = "",
                 schedule_options: Optional[Dict[str, Any]] = None,
                 advanced_options: Optional[Dict[str, Any]] = None,
                 hybrid_angle_threshold_deg: float = 15.0,
                 hybrid_smoothness: float = 0.1,
                 angleclamp_theta_max_deg: float = 45.0,
                 sbezier_tension: float = 0.5,
                 ease_kind: str = "smoothstep",
                 ease_gamma: float = 1.0):

        cond_in: List[Tuple[torch.Tensor, Dict[str, Any]]] = conditioning
        cond = flatten_conditioning(cond_in)

        needs_ref = mode in ("slerp", "bislerp", "lerp", "bilerp",
                              "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp")
        ref = None
        if needs_ref:
            if reference_conditioning is None:
                raise ValueError("reference_conditioning required for this mode")
            ref = get_reference_like(flatten_conditioning(reference_conditioning), cond)

        out: List[Tuple[torch.Tensor, Dict[str, Any]]] = []

        for i, (emb, opts) in enumerate(cond):
            base = emb
            B, T, D = emb.shape
            device, dtype = emb.device, emb.dtype

            # Schedule overrides
            sched_ref_s = ref_schedule
            sched_scale_s = scale_schedule
            sched_blend_s = blend_schedule
            if isinstance(schedule_options, dict):
                sched_ref_s = schedule_options.get("ref_schedule", sched_ref_s)
                sched_scale_s = schedule_options.get("scale_schedule", sched_scale_s)
                sched_blend_s = schedule_options.get("blend_schedule", sched_blend_s)

            ref_sched_vec = build_schedule(sched_ref_s, T, device, dtype)
            scale_sched_vec = build_schedule(sched_scale_s, T, device, dtype)
            blend_sched_vec = build_schedule(sched_blend_s, T, device, dtype)

            t_ref_scalar = apply_scalar_or_schedule_mean(ref_sched_vec, t_ref, device, dtype)
            blend_scalar = apply_scalar_or_schedule_mean(blend_sched_vec, blend_back, device, dtype)
            scale_scalar = torch.as_tensor(scale, device=device, dtype=dtype)

            t_ref_vec = None
            if ref_sched_vec is not None:
                t_ref_vec = ref_sched_vec.view(1, T).expand(B, T)

            blend_vec = None
            if blend_sched_vec is not None:
                blend_vec = blend_sched_vec.view(1, T).expand(B, T)

            scale_vec = None
            if scale_sched_vec is not None:
                scale_vec = scale_sched_vec.view(1, T, 1)

            # Advanced options overrides
            adv = advanced_options or {}
            hybrid_angle_threshold_deg = float(adv.get("hybrid_angle_threshold_deg", hybrid_angle_threshold_deg))
            hybrid_smoothness = float(adv.get("hybrid_smoothness", hybrid_smoothness))
            angleclamp_theta_max_deg = float(adv.get("angleclamp_theta_max_deg", angleclamp_theta_max_deg))
            sbezier_tension = float(adv.get("sbezier_tension", sbezier_tension))
            ease_kind = str(adv.get("ease_kind", ease_kind))
            ease_gamma = float(adv.get("ease_gamma", ease_gamma))

            if mode == "scale":
                mod = scale_embeddings(emb, scale_vec if scale_vec is not None else scale_scalar)
                if (blend_vec is not None and torch.any(blend_vec > 0)) or float(blend_scalar) > 0.0:
                    t_blend = blend_vec if blend_vec is not None else blend_scalar
                    mod = phase_blend_back(mod, base, t_blend, blend_back_interp, keep_magnitude, eps)

            elif mode in ("slerp", "lerp"):
                ref_emb, _ = ref[i]
                emb_b, ref_b = match_token_shapes(emb, ref_emb)
                t_to_ref = t_ref_vec if t_ref_vec is not None else t_ref_scalar
                mod = phase_to_reference(
                    emb_b, ref_b, t_to_ref, scale_scalar, scale_vec, ref_interp, keep_magnitude, eps
                )

            elif mode in ("bislerp", "bilerp", "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp"):
                ref_emb, _ = ref[i]
                emb_b, ref_b = match_token_shapes(emb, ref_emb)
                t_to_ref = t_ref_vec if t_ref_vec is not None else t_ref_scalar
                y_to_ref = apply_bimode_kernel(
                    mode, emb_b, ref_b, t_to_ref,
                    hybrid_angle_threshold_deg=hybrid_angle_threshold_deg,
                    hybrid_smoothness=hybrid_smoothness,
                    angleclamp_theta_max_deg=angleclamp_theta_max_deg,
                    sbezier_tension=sbezier_tension,
                    ease_kind=ease_kind,
                    ease_gamma=ease_gamma,
                    eps=eps,
                )

                mod = y_to_ref
                if keep_magnitude:
                    mod = apply_keep_magnitude(emb_b, mod, eps=eps)
                if scale_vec is not None:
                    if scale_vec.ndim == 1:
                        scale_vec = scale_vec.view(1, -1, 1)
                    mod = emb_b + (mod - emb_b) * scale_vec
                elif float(scale_scalar) != 1.0:
                    mod = emb_b + (mod - emb_b) * scale_scalar
                t_blend = blend_vec if blend_vec is not None else blend_scalar
                if (blend_vec is not None and torch.any(blend_vec > 0)) or float(blend_scalar) > 0.0:
                    mod = apply_bimode_kernel(
                        mode, mod, base, t_blend,
                        hybrid_angle_threshold_deg=hybrid_angle_threshold_deg,
                        hybrid_smoothness=hybrid_smoothness,
                        angleclamp_theta_max_deg=angleclamp_theta_max_deg,
                        sbezier_tension=sbezier_tension,
                        ease_kind=ease_kind,
                        ease_gamma=ease_gamma,
                        eps=eps,
                    )
                    if keep_magnitude:
                        mod = apply_keep_magnitude(base, mod, eps=eps)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            mod = safe_device_dtype_like(mod, emb)

            def pooled_apply(x):
                pr_t = apply_scalar_or_schedule_mean(ref_sched_vec, t_ref, x.device, x.dtype)
                pb_t = apply_scalar_or_schedule_mean(blend_sched_vec, blend_back, x.device, x.dtype)
                ps = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
                if scale_sched_vec is not None:
                    ps = torch.as_tensor(float(scale_sched_vec.mean().item()), device=x.device, dtype=x.dtype)

                if mode == "scale":
                    y = x * ps
                    if float(pb_t) > 0.0:
                        y = interp(y, x, pb_t, blend_back_interp, eps=eps)
                        if keep_magnitude:
                            y = apply_keep_magnitude(x, y, eps=eps)
                    return y

                if ref is not None and "pooled_output" in ref[i][1]:
                    r = ref[i][1]["pooled_output"].to(device=x.device, dtype=x.dtype)
                    if mode in ("bislerp", "bilerp", "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp"):
                        y = apply_bimode_kernel(
                            mode, x, r, pr_t,
                            hybrid_angle_threshold_deg=hybrid_angle_threshold_deg,
                            hybrid_smoothness=hybrid_smoothness,
                            angleclamp_theta_max_deg=angleclamp_theta_max_deg,
                            sbezier_tension=sbezier_tension,
                            ease_kind=ease_kind,
                            ease_gamma=ease_gamma,
                            eps=eps,
                        )
                    else:
                        y = interp(x, r, pr_t, ref_interp, eps=eps)
                    if keep_magnitude:
                        y = apply_keep_magnitude(x, y, eps=eps)
                    if float(ps) != 1.0:
                        y = x + (y - x) * ps
                    if mode in ("bislerp", "bilerp", "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp") and float(pb_t) > 0.0:
                        y = apply_bimode_kernel(
                            mode, y, x, pb_t,
                            hybrid_angle_threshold_deg=hybrid_angle_threshold_deg,
                            hybrid_smoothness=hybrid_smoothness,
                            angleclamp_theta_max_deg=angleclamp_theta_max_deg,
                            sbezier_tension=sbezier_tension,
                            ease_kind=ease_kind,
                            ease_gamma=ease_gamma,
                            eps=eps,
                        )
                        if keep_magnitude:
                            y = apply_keep_magnitude(x, y, eps=eps)
                    return y
                return x

            maybe_mod_pooled(opts, pooled_apply)
            out.append((mod, opts))

        return (out,)


class WASABI_CLIPTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "`CLIP` or `text_encoder` model(s)."}),
                "text": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": "Prompt to encode."
                }),
                "scale": ("FLOAT", {
                    "default": 1.50, "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": "Global scale used to scale conditioning embeddings."
                }),
                "mode": ([
                    "bislerp", "bilerp", "scale",
                    "binlerp", "bihybrid", "biangleclamp", "bisbezier", "biease_slerp", "biease_lerp"
                ], {
                    "default": "bislerp",
                    "tooltip": "Interpolation mode for conditioning embeddings."
                }),
            },
            "optional": {
                "advanced_options": ("DICT", {
                    "tooltip": "Optional advanced interpolation parameters produced by WASABI AdvancedOptions node."
                }),
                "schedule_options": ("DICT", {
                    "tooltip": "Optional schedules bundle from WASABI ScheduleOptions node."
                })
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    
    FUNCTION = "encode"
    CATEGORY = "for_testing"

    def encode(self, clip, text: str, scale: float, mode: str, advanced_options: Dict[str, Any] = None, schedule_options: Dict[str, Any] = None):
        base_encoder = CLIPTextEncode()
        (conditioning,) = base_encoder.encode(clip=clip, text=text)

        mod = WASABI_ConditioningModulate()
        opts = advanced_options or {}
        hybrid_angle_threshold_deg = float(opts.get("hybrid_angle_threshold_deg", 15.0))
        hybrid_smoothness = float(opts.get("hybrid_smoothness", 0.1))
        angleclamp_theta_max_deg = float(opts.get("angleclamp_theta_max_deg", 45.0))
        sbezier_tension = float(opts.get("sbezier_tension", 0.5))
        ease_kind = str(opts.get("ease_kind", "smoothstep"))
        ease_gamma = float(opts.get("ease_gamma", 1.0))
        (out,) = mod.modulate(
            conditioning=conditioning,
            reference_conditioning=conditioning,
            mode=mode,
            scale=scale,
            blend_back=0.50,
            keep_magnitude=True,
            eps=1e-8,
            t_ref=0.50,
            ref_interp="slerp",
            blend_back_interp="slerp",
            ref_schedule="",
            scale_schedule="",
            blend_schedule="",
            schedule_options=schedule_options,
            hybrid_angle_threshold_deg=hybrid_angle_threshold_deg,
            hybrid_smoothness=hybrid_smoothness,
            angleclamp_theta_max_deg=angleclamp_theta_max_deg,
            sbezier_tension=sbezier_tension,
            ease_kind=ease_kind,
            ease_gamma=ease_gamma,
        )
        return (out,)


class WASABI_ScheduleOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "token_length": ("INT", {
                    "default": 77, 
                    "min": 1, 
                    "max": 4096, 
                    "step": 1, 
                    "tooltip": "Token length for plot display (doesn't need to match input conditioning)."
                }),
                "ref_schedule": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Schedule for reference embeddings."
                }),
                "scale_schedule": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Schedule for scale."
                }),
                "blend_schedule": ("STRING", {
                    "default": "", 
                    "multiline": True, 
                    "tooltip": "Schedule for blend-back."
                }),
                "t_ref": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "Global fraction toward reference when ref_schedule is empty."
                }),
                "scale": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.01, 
                    "tooltip": "Global scale when no scale_schedule."
                }),
                "blend_back": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01, 
                    "tooltip": "Amount to blend results back toward ORIGINAL (0–1). Overridden per-token by blend_schedule."}),
            }
        }

    RETURN_TYPES = ("DICT", "IMAGE")
    RETURN_NAMES = ("schedule_options", "plot")
    FUNCTION = "build"
    CATEGORY = "for_testing"

    def plot_curves(self, curves: List[torch.Tensor], colors: List[Tuple[float, float, float]], H: int = 256, W: int = 512, margin: int = 10) -> torch.Tensor:
        img = torch.ones(H, W, 3, dtype=torch.float32)

        # axes
        img[H - margin: H - margin + 1, margin:W - margin, :] = 0.85  # x-axis (token index)
        img[margin:H - margin, margin:margin + 1, :] = 0.85            # y-axis (amplitude)

        if curves:
            all_vals = torch.cat([c.flatten() for c in curves])
            vmin = float(all_vals.min().item())
            vmax = float(all_vals.max().item())
            if not torch.isfinite(torch.tensor([vmin, vmax])).all():
                vmin, vmax = 0.0, 1.0
            if abs(vmax - vmin) < 1e-6:
                vmin -= 0.5
                vmax += 0.5
        else:
            vmin, vmax = 0.0, 1.0

        def to_px(xi: int, yi: float, N: int):
            x = margin + int(round((W - 2 * margin - 1) * (xi / max(1, N - 1))))
            y = margin + int(round((H - 2 * margin - 1) * (1.0 - (yi - vmin) / (vmax - vmin))))
            return x, y

        # tick marks: bottom (token length) and left (amplitude)
        def vline(x: int, y0: int, y1: int, col=0.7):
            y0, y1 = sorted((y0, y1))
            img[y0:y1 + 1, x, 0] = col
            img[y0:y1 + 1, x, 1] = col
            img[y0:y1 + 1, x, 2] = col

        def hline(y: int, x0: int, x1: int, col=0.7):
            x0, x1 = sorted((x0, x1))
            img[y, x0:x1 + 1, 0] = col
            img[y, x0:x1 + 1, 1] = col
            img[y, x0:x1 + 1, 2] = col

        # bottom ticks at 0, mid, N-1 (will use longest curve length if multiple differ)
        N_plot = max((c.numel() for c in curves), default=1)
        for xi in [0, (N_plot - 1) // 2, N_plot - 1]:
            x, y = to_px(xi, vmin, N_plot)
            vline(x, y, min(y + 5, H - 1), col=0.7)

        # left ticks at vmin, mid, vmax
        for yi in [vmin, (vmin + vmax) * 0.5, vmax]:
            x, y = to_px(0, yi, 2)
            hline(y, max(0, x - 5), x, col=0.7)

        # simple legend squares in top-left
        legend_y = margin + 3
        legend_x = margin + 4
        for idx, col in enumerate(colors):
            y0 = legend_y + idx * 8
            x0 = legend_x
            img[y0:y0 + 5, x0:x0 + 8, 0] = col[0]
            img[y0:y0 + 5, x0:x0 + 8, 1] = col[1]
            img[y0:y0 + 5, x0:x0 + 8, 2] = col[2]

        for vec, col in zip(curves, colors):
            N = vec.numel()
            px_prev = None
            for i in range(N):
                x, y = to_px(i, float(vec[i].item()), N)
                if px_prev is not None:
                    x0, y0 = px_prev
                    steps = max(abs(x - x0), abs(y - y0)) + 1
                    for t in range(steps + 1):
                        xi = int(round(x0 + (x - x0) * (t / max(1, steps))))
                        yi = int(round(y0 + (y - y0) * (t / max(1, steps))))
                        if 0 <= yi < H and 0 <= xi < W:
                            img[yi, xi, 0] = col[0]
                            img[yi, xi, 1] = col[1]
                            img[yi, xi, 2] = col[2]
        # to BHWC RGB float (ComfyUI expects BHWC)
        return img.unsqueeze(0)

    def build(self, token_length: int, ref_schedule: str, scale_schedule: str, blend_schedule: str, t_ref: float, scale: float, blend_back: float):
        device = torch.device("cpu")
        dtype = torch.float32

        T = int(token_length)
        # Build vectors for plotting (expand scalars if schedule empty)
        ref_vec = build_schedule(ref_schedule, T, device, dtype)
        if ref_vec is None:
            ref_vec = torch.full((T,), float(t_ref), dtype=dtype, device=device)
        scale_vec = build_schedule(scale_schedule, T, device, dtype)
        if scale_vec is None:
            scale_vec = torch.full((T,), float(scale), dtype=dtype, device=device)
        blend_vec = build_schedule(blend_schedule, T, device, dtype)
        if blend_vec is None:
            blend_vec = torch.full((T,), float(blend_back), dtype=dtype, device=device)

        plot = self.plot_curves([ref_vec, scale_vec, blend_vec], colors=[(1.0, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.4, 1.0)])

        return ({
            "ref_schedule": ref_schedule,
            "scale_schedule": scale_schedule,
            "blend_schedule": blend_schedule,
            "T": int(T),
            "t_ref": float(t_ref),
            "scale": float(scale),
            "blend_back": float(blend_back),
        }, plot)


class WASABI_AdvancedOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hybrid_angle_threshold_deg": ("FLOAT", {
                    "default": 15.0, 
                    "min": 0.0, 
                    "max": 180.0, 
                    "step": 0.1,
                    "tooltip": "Angle threshold (deg) where hybrid switches from LERP to SLERP."}),
                "hybrid_smoothness": ("FLOAT", {
                    "default": 0.1, 
                    "min": 1e-6, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Smoothness (radians) of hybrid transition. Larger = softer."}),
                "angleclamp_theta_max_deg": ("FLOAT", {
                    "default": 45.0, 
                    "min": 0.0, 
                    "max": 180.0, 
                    "step": 0.1,
                    "tooltip": "Maximum rotation angle for angle-clamped SLERP (deg)."}),
                "sbezier_tension": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01,
                    "tooltip": "Spherical Bézier tension (0..1)."}),
                "ease_kind": (["smoothstep", "cosine", "gamma", "identity"], {
                    "default": "smoothstep",
                    "tooltip": "Easing/remap for t used by biease_* modes."
                }),
                "ease_gamma": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.01, 
                    "max": 10.0, 
                    "step": 0.01,
                    "tooltip": "Gamma exponent for 'gamma' ease kind."}),
            }
        }

    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "for_testing"

    def build(self,
              hybrid_angle_threshold_deg: float,
              hybrid_smoothness: float,
              angleclamp_theta_max_deg: float,
              sbezier_tension: float,
              ease_kind: str,
              ease_gamma: float):
        return ({
            "hybrid_angle_threshold_deg": float(hybrid_angle_threshold_deg),
            "hybrid_smoothness": float(hybrid_smoothness),
            "angleclamp_theta_max_deg": float(angleclamp_theta_max_deg),
            "sbezier_tension": float(sbezier_tension),
            "ease_kind": str(ease_kind),
            "ease_gamma": float(ease_gamma),
        },)


NODE_CLASS_MAPPINGS = {
    "WASABI_ConditioningModulate": WASABI_ConditioningModulate,
    "WASABI_CLIPTextEncode": WASABI_CLIPTextEncode,
    "WASABI_ScheduleOptions": WASABI_ScheduleOptions,
    "WASABI_AdvancedOptions": WASABI_AdvancedOptions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASABI_ConditioningModulate": "WASABI Conditioning Modulate",
    "WASABI_CLIPTextEncode": "WASABI CLIP Text Encode",
    "WASABI_ScheduleOptions": "WASABI Schedule Options",
    "WASABI_AdvancedOptions": "WASABI Advanced Options",
}
