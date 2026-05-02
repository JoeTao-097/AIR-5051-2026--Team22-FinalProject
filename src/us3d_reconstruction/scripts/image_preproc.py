#!/usr/bin/env python3
"""Ultrasound image preprocessing for higher-quality 3D reconstruction.

Each utility is independent; you can chain them in any order via
``apply_pipeline`` or use a single pre-built ``preprocess_for_pbm``
recipe.

Operations
──────────
``tgc_equalize``        Depth-wise gain compensation (per-row mean
                         normalization). Removes residual TGC bias so
                         deep tissue isn't permanently dimmer than
                         shallow tissue.

``despeckle``           Bilateral filter (edge-preserving denoising).

``shadow_weight``        Returns a (H, W) confidence map that downweights
                         acoustic shadows (regions immediately under a
                         strong reflector).

``gradient_weight``     Returns a (H, W) map proportional to local
                         edge energy — emphasizes echogenic structure
                         and de-emphasizes uniform speckle/background.

``geometry_mask``       Static binary mask for the cropped ROI (drops
                         the side fan and depth-marker columns).

``apply_pipeline``      Compose any subset on a single uint8 image.

``preprocess_for_pbm``   Recommended default recipe used by the new
                         pbm_compound_v2 pipeline.

``make_pixel_weight_fn`` Build a single ``f(img) → weight_map`` closure
                         for ``pbm_compound_v2``'s ``pixel_weight_fn``.
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Time-Gain Compensation equalization
# ─────────────────────────────────────────────────────────────────────────────

def apply_tgc_eq(img, intensity_min=15, gamma=0.5,
                 smoothing_window=21):
    """Compatibility shim: TGC equalization with ``gamma`` exponent.

    Equivalent to ``tgc_equalize(img, intensity_floor=intensity_min,
    strength=gamma, smoothing_window=smoothing_window)``.
    """
    return tgc_equalize(img, intensity_floor=intensity_min,
                        strength=gamma, smoothing_window=smoothing_window)


def tgc_equalize(img, target_mean=None, intensity_floor=15,
                 smoothing_window=21, strength=0.7):
    """Per-row mean normalization to flatten depth-dependent gain bias.

    ``target_mean`` defaults to the global mean of pixels above
    ``intensity_floor`` so we don't drag the background up. Each row's
    pixels are scaled by ``(target_mean / row_mean) ** strength``;
    ``strength=1.0`` is a full equalization, ``0.7`` is a softer
    version that preserves natural attenuation contrast.

    The row-mean profile is smoothed with a moving average to avoid
    amplifying single-row artifacts (cursor lines, scaler ticks).
    """
    img32 = img.astype(np.float32)
    h, w = img32.shape
    valid = img32 >= intensity_floor

    if target_mean is None:
        if valid.any():
            target_mean = float(img32[valid].mean())
        else:
            target_mean = float(img32.mean())

    # Per-row mean over valid pixels (zero where row is empty)
    row_count = valid.sum(axis=1).astype(np.float32)
    row_sum = (img32 * valid).sum(axis=1)
    row_mean = np.where(row_count > 0, row_sum / np.maximum(row_count, 1), 0)

    # Smooth row_mean
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window, dtype=np.float32) / smoothing_window
        # Replicate-pad to avoid edge dimming
        pad = smoothing_window // 2
        padded = np.pad(row_mean, pad, mode='edge')
        row_mean_s = np.convolve(padded, kernel, mode='valid')
    else:
        row_mean_s = row_mean

    eps = 1.0
    gain = np.where(row_mean_s > eps,
                    (target_mean / np.maximum(row_mean_s, eps)) ** strength,
                    1.0).astype(np.float32)
    # Clamp gain to avoid blowing up under-sampled rows
    gain = np.clip(gain, 0.5, 2.5)

    out = img32 * gain[:, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Despeckle
# ─────────────────────────────────────────────────────────────────────────────

def despeckle(img, method='bilateral', ksize=3,
              sigma_color=25, sigma_space=5):
    """Edge-preserving denoising. Default = small-radius bilateral.

    Args:
        method: 'bilateral' | 'median' | 'gauss' | 'nlmeans' | 'none'
        ksize: kernel size; for bilateral interpreted as diameter ``d``.
            For median: square filter size (odd).
            For gauss: passed to GaussianBlur (must be odd, >=1).
        sigma_color, sigma_space: bilateral-only photometric / spatial σ.

    For ultrasound, ``bilateral`` keeps fascia/bone edges intact while
    smoothing speckle inside soft tissue. ``median`` is faster but
    blurs fine structures. ``nlmeans`` is highest quality but ~10×
    slower.
    """
    if cv2 is None or method in (None, 'none'):
        return img
    if method == 'bilateral':
        d = max(int(ksize), 1)
        return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color,
                                    sigmaSpace=sigma_space)
    if method == 'median':
        k = int(ksize) | 1  # force odd
        return cv2.medianBlur(img, ksize=k)
    if method == 'gauss':
        k = int(ksize) | 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == 'nlmeans':
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7,
                                         searchWindowSize=21)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 3. Acoustic shadow weight
# ─────────────────────────────────────────────────────────────────────────────

def shadow_weight(img, intensity_floor=15,
                  bright_thr=180, falloff_rows=40, min_w=0.15):
    """Confidence map that downweights pixels behind strong reflectors.

    Strategy:
      - For each column, find rows of "very bright" pixels (intensity
        > ``bright_thr``).
      - Mark a falloff zone of ``falloff_rows`` rows immediately below
        them with reduced weight.
      - Weight transitions linearly from ``min_w`` at the brightest
        reflector to 1 at distance ``falloff_rows``.

    Output is float32 in [min_w, 1]. Pixels in the dark background
    (intensity < ``intensity_floor``) are forced to 0.
    """
    h, w = img.shape
    img32 = img.astype(np.float32)
    weight = np.ones_like(img32)

    bright_mask = img32 > bright_thr

    # Per-column cumulative max-over-window of "shadow strength"
    # We compute a per-column "rows-since-last-bright" map, capped at
    # falloff_rows. The weight is proportional to (rows / falloff_rows).
    rows_since = np.full((h, w), falloff_rows + 1, dtype=np.int32)
    last_bright_row = np.full(w, -10**6, dtype=np.int32)
    for r in range(h):
        cols_bright = np.where(bright_mask[r])[0]
        if len(cols_bright):
            last_bright_row[cols_bright] = r
        dist = r - last_bright_row
        rows_since[r] = np.minimum(dist, falloff_rows + 1)

    # Linear ramp: dist=0 → min_w, dist≥falloff → 1
    w_ramp = (np.minimum(rows_since, falloff_rows).astype(np.float32) /
              max(falloff_rows, 1))
    weight = min_w + (1.0 - min_w) * w_ramp

    # Hard zero on background
    weight = np.where(img32 < intensity_floor, 0.0, weight).astype(np.float32)
    return weight


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient-based confidence
# ─────────────────────────────────────────────────────────────────────────────

def edge_weight(img, sigma=1.0, scale=20.0, base=0.3):
    """Alias for ``gradient_weight`` (older name kept for compat)."""
    return gradient_weight(img, sigma=sigma, scale=scale, base=base)


def gradient_weight(img, sigma=1.0, scale=20.0, base=0.3):
    """Boost weight where edges are strong; soft-floor for flat regions.

    weight(p) = base + (1 - base) * sigmoid((|∇I| - scale) / scale)

    With ``base=0.3``, flat areas still contribute (preserves tissue
    average) but echogenic boundaries dominate.
    """
    if cv2 is not None:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    else:
        f = img.astype(np.float32)
        gx = np.zeros_like(f); gy = np.zeros_like(f)
        gx[:, 1:-1] = 0.5 * (f[:, 2:] - f[:, :-2])
        gy[1:-1, :] = 0.5 * (f[2:, :] - f[:-2, :])
    g = np.sqrt(gx * gx + gy * gy)
    # Smooth slightly to spread credit to neighboring pixels
    if cv2 is not None and sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    s = 1.0 / (1.0 + np.exp(-(g - scale) / max(scale, 1.0)))
    return (base + (1.0 - base) * s).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Static geometry mask (no-op default — ROI already crops UI)
# ─────────────────────────────────────────────────────────────────────────────

def geometry_mask(img_shape, side_margin=0, top_margin=0, bottom_margin=0):
    """Optional rectangular mask in case the ROI still leaks UI."""
    h, w = img_shape
    m = np.ones((h, w), dtype=np.float32)
    if side_margin > 0:
        m[:, :side_margin] = 0
        m[:, -side_margin:] = 0
    if top_margin > 0:
        m[:top_margin, :] = 0
    if bottom_margin > 0:
        m[-bottom_margin:, :] = 0
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Composers
# ─────────────────────────────────────────────────────────────────────────────

def apply_pipeline(img, ops=()):
    """Apply a sequence of (name, kwargs) image transforms.

    Recognized names: 'tgc', 'despeckle'. Other names ignored.
    """
    out = img
    for name, kw in ops:
        if name == 'tgc':
            out = tgc_equalize(out, **kw)
        elif name == 'despeckle':
            out = despeckle(out, **kw)
    return out


def compose_xform(*fns):
    """Return ``f(img) → img`` that applies ``fn0, fn1, ...`` in order."""
    def composed(img):
        out = img
        for fn in fns:
            out = fn(out)
        return out
    return composed


def compose_weight(*fns):
    """Return ``f(img) → (H, W) float32`` = product of ``fn(img)`` maps.

    Each ``fn`` must accept the (transformed) image and return a
    (H, W) float32 weight array.
    """
    def composed(img):
        h, w = img.shape
        out = np.ones((h, w), dtype=np.float32)
        for fn in fns:
            out = out * fn(img).astype(np.float32)
        return out
    return composed


def transform_frames(frames, fn):
    """Apply ``fn`` to every frame, returning a new list. ``fn`` should
    accept a uint8 (H, W) image and return a uint8 (H, W) image."""
    return [fn(im) for im in frames]


def preprocess_for_pbm(img, do_tgc=True, do_despeckle=True,
                       tgc_strength=0.7, despeckle_method='bilateral'):
    """Recommended image transform applied BEFORE splatting."""
    out = img
    if do_tgc:
        out = tgc_equalize(out, strength=tgc_strength)
    if do_despeckle:
        out = despeckle(out, method=despeckle_method)
    return out


def make_pixel_weight_fn(do_shadow=True, do_gradient=True,
                         shadow_kwargs=None, gradient_kwargs=None,
                         static_mask_kwargs=None):
    """Build a closure ``f(img) → (H, W) float32 weight``.

    Use the returned closure as ``pixel_weight_fn`` for
    ``pbm_compound_v2``. The image passed to ``f`` is the *post-
    preprocessing* image (so it sees the same TGC/despeckled values
    that get splatted).
    """
    shadow_kwargs = shadow_kwargs or {}
    gradient_kwargs = gradient_kwargs or {}
    static_mask_kwargs = static_mask_kwargs or {}

    def fn(img):
        h, w = img.shape
        wt = np.ones((h, w), dtype=np.float32)
        if static_mask_kwargs:
            wt *= geometry_mask((h, w), **static_mask_kwargs)
        if do_shadow:
            wt *= shadow_weight(img, **shadow_kwargs)
        if do_gradient:
            wt *= gradient_weight(img, **gradient_kwargs)
        return wt
    return fn
