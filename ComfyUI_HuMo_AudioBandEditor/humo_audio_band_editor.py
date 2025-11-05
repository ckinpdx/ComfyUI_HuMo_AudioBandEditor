
import torch

class HuMoAudioBandEditor:
    """
    Edit HuMo audio embeddings in-place:
      - Per-band gains for the 5 Whisper-derived bands
      - Temporal EMA smoothing (to reduce flicker)
      - Optional RMS preservation (keeps scale stable)
      - Final mix with original via alpha (residual blend)
      - Optional clamp
    Expects embeds dict with key 'humo_audio_emb' of shape [T, 5, 1280].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                # Per-band gains (feat0..feat4)
                "gain_b0": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Band 0 (shallow edges/onsets)"}),
                "gain_b1": ("FLOAT", {"default": 1.20, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Band 1 (short-term rhythm)"}),
                "gain_b2": ("FLOAT", {"default": 1.10, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Band 2 (phrase patterns)"}),
                "gain_b3": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Band 3 (long-range cadence)"}),
                "gain_b4": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Band 4 (top semantic)"}),
                # Temporal smoothing & blending
                "ema_beta": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 0.999, "step": 0.001, "tooltip": "EMA over time; higher = smoother"}),
                "preserve_rms": ("BOOLEAN", {"default": True, "tooltip": "Match per-frame, per-band RMS to original"}),
                "alpha_mix": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Residual blend with original (0 keeps original, 1 uses fully edited)"}),
                "global_gain": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Post-mix global gain"}),
                # Safety clamp (0 disables)
                "clamp_std": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Clip to mean±N·std; 0 to disable"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "apply"
    CATEGORY = "HuMo Audio/Motion"
    DESCRIPTION = "Band-gain editor for HuMo audio embeddings (T,5,1280)."

    def _rms(self, x, eps=1e-6):
        # per-frame, per-band RMS (keep channel dim)
        return x.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)

    def apply(self, image_embeds,
              gain_b0, gain_b1, gain_b2, gain_b3, gain_b4,
              ema_beta, preserve_rms, alpha_mix, global_gain, clamp_std):

        # shallow copy dict to avoid side-effects on other references
        embeds = dict(image_embeds)
        key = "humo_audio_emb"
        if key not in embeds:
            raise ValueError("Missing key 'humo_audio_emb' in WANVIDIMAGE_EMBEDS")

        x = embeds[key]  # [T, 5, 1280]
        if x.ndim != 3 or x.shape[1] != 5:
            raise ValueError(f"humo_audio_emb expected shape [T,5,C], got {tuple(x.shape)}")

        device = x.device
        dtype = x.dtype

        x_orig = x

        # Per-band gains
        gains = torch.tensor([gain_b0, gain_b1, gain_b2, gain_b3, gain_b4], device=device, dtype=dtype).view(1, 5, 1)

        # Temporal EMA smoothing (bandwise on channels)
        if x.shape[0] > 1 and ema_beta > 0.0:
            s = x[0].clone()
            out_seq = [s]
            beta = torch.tensor(ema_beta, device=device, dtype=dtype)
            one_m = (1.0 - beta).to(dtype)
            for t in range(1, x.shape[0]):
                s = beta * s + one_m * x[t]
                out_seq.append(s)
            x_smooth = torch.stack(out_seq, dim=0)
        else:
            x_smooth = x

        # Apply per-band gains
        x_edit = x_smooth * gains

        # Optional RMS preservation (per-frame, per-band)
        if preserve_rms:
            rms_orig = self._rms(x_orig)
            rms_edit = self._rms(x_edit)
            x_edit = x_edit * (rms_orig / rms_edit)

        # Residual blend + global gain
        alpha = torch.tensor(alpha_mix, device=device, dtype=dtype)
        x_mixed = (1.0 - alpha) * x_orig + alpha * x_edit
        if global_gain != 1.0:
            x_mixed = x_mixed * global_gain

        # Optional clamp to curb snow
        if clamp_std > 0.0:
            mean = x_mixed.mean()
            std = x_mixed.std().clamp_min(1e-6)
            lo = mean - clamp_std * std
            hi = mean + clamp_std * std
            x_mixed = x_mixed.clamp(lo, hi)

        embeds[key] = x_mixed
        return (embeds,)

NODE_CLASS_MAPPINGS = {
    "HuMoAudioBandEditor": HuMoAudioBandEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HuMoAudioBandEditor": "HuMo Audio Band Editor",
}
