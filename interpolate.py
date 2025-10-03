import math
import torch


def unit_vector(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim < a.ndim:
        t = t.unsqueeze(-1)
    return a + (b - a) * t


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_u = unit_vector(a, eps)
    b_u = unit_vector(b, eps)
    dot = (a_u * b_u).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.arccos(dot)
    sin_omega = torch.sin(omega).clamp_min(eps)
    if t.ndim < a.ndim:
        t = t.unsqueeze(-1)
    w1 = torch.sin((1.0 - t) * omega) / sin_omega
    w2 = torch.sin(t * omega) / sin_omega
    return w1 * a + w2 * b


def nlerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized LERP (cheap spherical-ish)."""
    y = lerp(a, b, t)
    return unit_vector(y, eps)


def t_gamma(t: torch.Tensor, gamma: float) -> torch.Tensor:
    if gamma == 1.0:
        return t
    return torch.clamp(t, 0.0, 1.0) ** gamma


def t_smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def t_cosine_ease(t: torch.Tensor) -> torch.Tensor:
    t = torch.clamp(t, 0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * t))


def remap_t(t: torch.Tensor, kind: str = "identity", gamma: float = 1.0) -> torch.Tensor:
    kind = (kind or "identity").lower()
    if kind == "identity":
        return t
    if kind == "gamma":
        return t_gamma(t, gamma)
    if kind == "smoothstep":
        return t_smoothstep(t)
    if kind == "cosine":
        return t_cosine_ease(t)
    return t


def hybrid_slerp_lerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    angle_threshold_deg: float = 15.0,
    smoothness: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Angle-aware hybrid: smoothly mixes LERP for small angles and SLERP for large angles.
    - angle_threshold_deg: angle where mix is ~0.5
    - smoothness: larger -> smoother/softer transition (in radians)
    """
    a_u = unit_vector(a, eps)
    b_u = unit_vector(b, eps)
    dot = (a_u * b_u).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot)
    th = math.radians(angle_threshold_deg)
    k = max(smoothness, 1e-6)
    # Logistic around threshold
    w = torch.sigmoid((theta - th) / k)
    y_l = lerp(a, b, t)
    y_s = slerp(a, b, t, eps=eps)
    return w * y_s + (1.0 - w) * y_l


def angle_clamped_slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    theta_max_deg: float = 45.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """SLERP but cap the effective rotation by theta_max_deg."""
    a_u = unit_vector(a, eps)
    b_u = unit_vector(b, eps)
    dot = (a_u * b_u).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.arccos(dot).clamp_min(1e-12)
    theta_max = math.radians(theta_max_deg)
    if t.ndim < a.ndim:
        t = t.unsqueeze(-1)
    # scale t so that t=1 corresponds to min(theta, theta_max)
    scale = torch.clamp(theta_max / theta, max=1.0)
    t_eff = torch.clamp(t * scale, 0.0, 1.0)
    return slerp(a, b, t_eff, eps=eps)


def spherical_bezier(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    tension: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Spherical cubic Bézier via chained SLERPs (de Casteljau on the sphere).
    Control points are placed along the great-circle using a simple tension scheme.
    """
    s = float(torch.clamp(torch.as_tensor(tension), 0.0, 1.0).item())
    # Controls closer to endpoints as tension increases
    c1 = slerp(a, b, torch.as_tensor(s * 0.5, device=a.device, dtype=a.dtype), eps=eps)
    c2 = slerp(a, b, torch.as_tensor(1.0 - s * 0.5, device=a.device, dtype=a.dtype), eps=eps)

    # de Casteljau on sphere
    p01 = slerp(a, c1, t, eps=eps)
    p12 = slerp(c1, c2, t, eps=eps)
    p23 = slerp(c2, b, t, eps=eps)
    p012 = slerp(p01, p12, t, eps=eps)
    p123 = slerp(p12, p23, t, eps=eps)
    return slerp(p012, p123, t, eps=eps)


def cosine_space(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Interpolate on the unit sphere by linear blending in vector space then renormalizing.
    This is effectively NLERP; provided separately for API clarity.
    """
    return nlerp(a, b, t, eps=eps)


def eased_interp(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    base: str = "slerp",
    ease: str = "smoothstep",
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply an easing/remapping to t, then interpolate with base kind (slerp/lerp/nlerp)."""
    t2 = remap_t(t, kind=ease, gamma=gamma)
    base = base.lower()
    if base == "slerp":
        return slerp(a, b, t2, eps=eps)
    if base == "nlerp":
        return nlerp(a, b, t2, eps=eps)
    return lerp(a, b, t2)


def orthogonal_component_mix(
    a: torch.Tensor,
    b: torch.Tensor,
    t: torch.Tensor,
    s_parallel: float = 1.0,
    s_orthogonal: float = 1.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Decompose displacement into parallel and orthogonal components relative to a, and mix with different gains.
    """
    if t.ndim < a.ndim:
        t = t.unsqueeze(-1)
    a_u = unit_vector(a, eps)
    d = b - a
    par = (d * a_u).sum(dim=-1, keepdim=True) * a_u
    orth = d - par
    y = a + (s_parallel * par + s_orthogonal * orth) * t
    if normalize:
        y = unit_vector(y, eps)
    return y
