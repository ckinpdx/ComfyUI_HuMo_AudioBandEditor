
# ComfyUI HuMo Audio Band Editor

Edit the HuMo audio embeddings (`humo_audio_emb`, shape `[T,5,1280]`) produced by **HuMo Embeds** before they enter the WAN video model.

## Features
- Per-band gains for the 5 Whisper-derived bands (0..4)
- Temporal EMA smoothing
- Optional RMS preservation (keeps scale stable)
- Residual blend with the original
- Optional clamp to reduce 'snow'

## Install
1. Unzip this folder into your ComfyUI custom nodes directory, e.g.:
   - `ComfyUI/custom_nodes/ComfyUI_HuMo_AudioBandEditor/`
2. Restart ComfyUI.

## Use
Place **HuMo Audio Band Editor** *right after* **HuMo Embeds**.
Connect its output to the rest of your HuMo/WAN pipeline (wherever you used the original `WANVIDIMAGE_EMBEDS`).

### Suggested starting values
- gain_b0 = 1.25, gain_b1 = 1.20, gain_b2 = 1.10, gain_b3 = 1.00, gain_b4 = 0.85
- ema_beta = 0.90
- preserve_rms = True
- alpha_mix = 0.35
- global_gain = 1.00
- clamp_std = 3.0

Lower `alpha_mix` or increase `ema_beta` if you see sparkly artifacts.
