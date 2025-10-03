import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional

from .interpolate import (
    lerp,
    slerp,
    nlerp,
    hybrid_slerp_lerp,
    angle_clamped_slerp,
    spherical_bezier,
     eased_interp,
)


MAX_SCHEDULE_STR_LEN = 16384
MAX_SCHEDULE_LIST_LEN = 8192
MAX_ABS_VAL = 64.0


def safe_device_dtype_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.to(device=ref.device, dtype=ref.dtype)


def unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def match_token_shapes(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xb, xt, xd = x.shape
    yb, yt, yd = y.shape
    if xd != yd:
        raise ValueError(f"Embedding dim mismatch: {xd} vs {yd}")
    if yb == 1 and xb > 1:
        y = y.expand(xb, yt, yd)
    elif xb == 1 and yb > 1:
        x = x.expand(yb, xt, xd)
    if xt != yt:
        if xt == 1 and yt > 1:
            x = x.expand(x.shape[0], yt, xd)
        elif yt == 1 and xt > 1:
            y = y.expand(y.shape[0], xt, yd)
        else:
            raise ValueError(f"Token length mismatch: {xt} vs {yt}. Use a reference with T==1 or matching T.")
    return x, y


def apply_keep_magnitude(original: torch.Tensor, modified: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True).clamp_min(eps)
    mod_unit = unit_vector(modified, eps)
    return mod_unit * orig_norm


def scale_embeddings(emb: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(scale):
        return emb * scale
    if scale.ndim < emb.ndim:
        scale = scale.unsqueeze(-1)
    return emb * scale


def maybe_mod_pooled(options: Dict[str, Any], f):
    if isinstance(options, dict) and "pooled_output" in options \
        and isinstance(options["pooled_output"], torch.Tensor):
        options["pooled_output"] = f(options["pooled_output"])


def parse_float_str(val: str) -> float:
    try:
        v = float(val)
    except Exception:
        raise ValueError(
            f"Expected numeric value in schedule, got '{val}'. "
            "Use numeric constants like linspace(0,1), cosine(0,1), flat(0.5), or a numeric list."
        )
    if not math.isfinite(v):
        raise ValueError("Non-finite value in schedule")
    if abs(v) > MAX_ABS_VAL:
        raise ValueError(f"Value out of allowed range ±{MAX_ABS_VAL}")
    return v


def parse_ab(inner: str) -> Tuple[float, float]:
    inner = inner.replace("->", ",")
    parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
    if len(parts) < 2:
        raise ValueError("Schedule requires two numeric values, e.g., linspace(0,1), cosine(0,1), or flat(0.5)")
    return parse_float_str(parts[0]), parse_float_str(parts[1])


def to_tensor_list(spec: str) -> List[float]:
    if len(spec) > MAX_SCHEDULE_STR_LEN:
        raise ValueError("Schedule string too long")
    s = spec
    if s.startswith("csv:"):
        s = s[4:]
    s = s.replace(",", " ")
    vals = [v for v in s.split() if v]
    if len(vals) > MAX_SCHEDULE_LIST_LEN:
        raise ValueError("Schedule list too long")
    out = []
    for v in vals:
        out.append(parse_float_str(v))
    return out


def resample_vec(vec: torch.Tensor, T: int, device, dtype) -> torch.Tensor:
    n = vec.numel()
    if n == T:
        y = vec.to(device=device, dtype=dtype)
    elif n == 1:
        y = vec.to(device=device, dtype=dtype).repeat(T)
    else:
        if n > MAX_SCHEDULE_LIST_LEN:
            raise ValueError("Schedule list too long")
        x = vec.view(1, 1, n).to(device=device, dtype=dtype)
        y = torch.nn.functional.interpolate(x, size=T, mode="linear", align_corners=True).view(T)
    if not torch.isfinite(y).all():
        raise ValueError("Schedule produced non-finite values")
    return y


def build_schedule(spec: str, T: int, device, dtype) -> Optional[torch.Tensor]:
    if spec is None:
        return None
    s = spec.strip()
    if s == "":
        return None
    s_low = s.lower()
    if s_low.startswith("linspace(") and s_low.endswith(")"):
        a, b = parse_ab(s[9:-1])
        w = torch.linspace(a, b, T, device=device, dtype=dtype)
        if not torch.isfinite(w).all():
            raise ValueError("linspace schedule non-finite")
        return w
    if s_low.startswith("cosine(") and s_low.endswith(")"):
        inner = s[7:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip() != ""]
        if len(parts) < 2:
            raise ValueError("cosine expects at least 2 numeric values, e.g., cosine(0,1) or cosine(0.2,0.2,0.8)")
        ctrl = [parse_float_str(p) for p in parts]
        M = len(ctrl)
        t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
        if M == 2:
            e = 0.5 * (1.0 - torch.cos(math.pi * t))
            y = ctrl[0] + (ctrl[1] - ctrl[0]) * e
        else:
            k = torch.clamp((t * (M - 1)).floor().to(torch.int64), 0, M - 2)
            t0 = k.to(t.dtype) / float(M - 1)
            u = (t - t0) * float(M - 1)
            u = torch.clamp(u, 0.0, 1.0)
            e = 0.5 * (1.0 - torch.cos(math.pi * u))
            v0 = torch.tensor([ctrl[i] for i in range(M - 1)], device=device, dtype=dtype)[k]
            v1 = torch.tensor([ctrl[i + 1] for i in range(M - 1)], device=device, dtype=dtype)[k]
            y = v0 + (v1 - v0) * e
        if not torch.isfinite(y).all():
            raise ValueError("cosine schedule non-finite")
        return y
    if s_low.startswith("flat(") and s_low.endswith(")"):
        v = parse_float_str(s[5:-1])
        return torch.full((T,), v, device=device, dtype=dtype)
    vals = to_tensor_list(s)
    if len(vals) == 0:
        return None
    vec = torch.tensor(vals)
    return resample_vec(vec, T, device, dtype)


def apply_scalar_or_schedule_mean(sched: Optional[torch.Tensor], fallback: float, device, dtype) -> torch.Tensor:
    if sched is None:
        v = float(fallback)
        if not math.isfinite(v) or abs(v) > MAX_ABS_VAL:
            raise ValueError("Fallback scalar non-finite or out of range")
        return torch.as_tensor(v, device=device, dtype=dtype)
    m = float(sched.mean().item())
    if not math.isfinite(m) or abs(m) > MAX_ABS_VAL:
        raise ValueError("Schedule mean non-finite or out of range")
    return torch.as_tensor(m, device=device, dtype=dtype)


def interp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, kind: str, eps: float = 1e-8) -> torch.Tensor:
    if kind == "slerp":
        return slerp(a, b, t, eps=eps)
    return lerp(a, b, t)


def apply_bimode_kernel(mode: str, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, *, 
    hybrid_angle_threshold_deg: float, hybrid_smoothness: float, angleclamp_theta_max_deg: 
    float, sbezier_tension: float, ease_kind: str, ease_gamma: float, eps: float) -> torch.Tensor:
    
    if mode == "bislerp":
        return slerp(a, b, t, eps=eps)
    if mode == "bilerp":
        return lerp(a, b, t)
    if mode == "binlerp":
        return nlerp(a, b, t, eps=eps)
    if mode == "bihybrid":
        return hybrid_slerp_lerp(a, b, t,
                                 angle_threshold_deg=float(hybrid_angle_threshold_deg),
                                 smoothness=float(hybrid_smoothness), eps=eps)
    if mode == "biangleclamp":
        return angle_clamped_slerp(a, b, t, theta_max_deg=float(angleclamp_theta_max_deg), eps=eps)
    if mode == "bisbezier":
        return spherical_bezier(a, b, t, tension=float(sbezier_tension), eps=eps)
    if mode == "biease_slerp":
        return eased_interp(a, b, t, base="slerp", ease=str(ease_kind), gamma=float(ease_gamma), eps=eps)
    if mode == "biease_lerp":
        return eased_interp(a, b, t, base="lerp", ease=str(ease_kind), gamma=float(ease_gamma), eps=eps)
    raise ValueError(f"Unsupported mode: {mode}")


def phase_to_reference(emb: torch.Tensor, ref_emb: torch.Tensor, t_ref: torch.Tensor, scale_scalar: torch.Tensor, 
    scale_vec: Optional[torch.Tensor], interp_kind: str, keep_magnitude: bool, eps: float) -> torch.Tensor:

    y = interp(emb, ref_emb, t_ref, interp_kind, eps=eps)
    if keep_magnitude:
        y = apply_keep_magnitude(emb, y, eps=eps)
    if scale_vec is not None:
        if scale_vec.ndim == 1:
            scale_vec = scale_vec.view(1, -1, 1)
        y = emb + (y - emb) * scale_vec
    elif float(scale_scalar) != 1.0:
        y = emb + (y - emb) * scale_scalar
    return y


def phase_blend_back(cur: torch.Tensor, base: torch.Tensor, blend_t: torch.Tensor, interp_kind: str, 
    keep_magnitude: bool, eps: float) -> torch.Tensor:

    y = interp(cur, base, blend_t, interp_kind, eps=eps)
    if keep_magnitude:
        y = apply_keep_magnitude(base, y, eps=eps)
    return y


def flatten_conditioning(cond: List[Tuple[torch.Tensor, Dict[str, Any]]]):
    out = []
    for c, o in cond:
        if not isinstance(c, torch.Tensor):
            raise TypeError("Conditioning tensor must be a torch.Tensor")
        if c.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(c.shape)}")
        out.append([c.clone(), dict(o) if isinstance(o, dict) else {}])
    return out


def get_reference_like(ref: Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]], 
    like: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Optional[List[Tuple[torch.Tensor, Dict[str, Any]]]]:
    
    if ref is None:
        return None
    if len(ref) == 1 and len(like) > 1:
        return [[ref[0][0], ref[0][1]] for _ in range(len(like))]
    if len(ref) != len(like):
        return [[ref[0][0], ref[0][1]] for _ in range(len(like))]
    return ref
