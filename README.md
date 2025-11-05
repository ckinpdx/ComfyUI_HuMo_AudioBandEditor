# HuMo Audio Band Editor

**⚠️ EXPERIMENTAL NODE - Use at your own risk**

ComfyUI custom node for editing HuMo audio embeddings with per-band gain control, temporal smoothing, and RMS preservation.

## What it does

This node modifies Whisper audio embeddings before they enter HuMo's AudioProjModel. It provides independent gain controls for each of Whisper's 5 audio feature bands:

- **Band 0** (feat0) - Shallow edges/onsets
- **Band 1** (feat1) - Short-term rhythm  
- **Band 2** (feat2) - Phrase patterns
- **Band 3** (feat3) - Long-range cadence
- **Band 4** (feat4) - Top semantic layer

By adjusting these bands independently, you can emphasize or suppress different audio characteristics. For example, boosting bands 0-2 while suppressing 3-4 can create beat-responsive motion while reducing phoneme/speech influence.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ckinpdx/ComfyUI_HuMo_AudioBandEditor
```
Restart ComfyUI.

## Usage

Place this node between HuMoEmbeds and the rest of your workflow:

```
WanVideo Model Loader → HuMoEmbeds → HuMo Audio Band Editor → Sampler → ...
```

The node takes `WANVIDIMAGE_EMBEDS` as input and returns modified embeddings.

## Parameters

### Per-band gains (0.0 - 4.0)
**gain_b0** (default: 1.25) - Band 0 gain  
**gain_b1** (default: 1.20) - Band 1 gain  
**gain_b2** (default: 1.10) - Band 2 gain  
**gain_b3** (default: 1.00) - Band 3 gain  
**gain_b4** (default: 0.85) - Band 4 gain

### Processing controls
**ema_beta** (default: 0.90, range: 0.0-0.999)
- Exponential moving average for temporal smoothing
- Higher values = smoother over time (reduces flicker)
- 0.0 = no smoothing (max responsiveness)

**preserve_rms** (default: True)
- Match per-frame, per-band RMS to original
- Keeps signal scale stable, prevents blow-out

**alpha_mix** (default: 0.35, range: 0.0-1.0)
- Residual blend between original and edited embeddings
- 0.0 = keep original (no effect)
- 1.0 = use fully edited signal

**global_gain** (default: 1.00, range: 0.0-4.0)
- Overall gain applied after mixing
- Post-processing amplification

**clamp_std** (default: 3.0, range: 0.0-10.0)
- Clips values to mean ± N·std
- Safety measure to prevent extreme values
- 0.0 = disabled

## Important Limitations

**⚠️ Experimental**
- No stable parameter values have been established
- Effects may vary depending on audio content
- May cause unexpected visual artifacts

**⚠️ Band interpretation is approximate**
- Whisper bands are trained on speech
- Band characteristics for music/non-speech audio are less predictable
- Experimentation required to find useful settings

**⚠️ May require other amplification**
- Band editing alone may not create strong motion response
- Often needs to be combined with other techniques for noticeable effects

## Technical Details

The node operates on `humo_audio_emb` tensors of shape `[T, 5, 1280]` where:
- T = number of frames
- 5 = Whisper's feature bands (layers 0-8, 8-16, 16-24, 24-32, 32)
- 1280 = embedding dimensions

Processing pipeline:
1. Apply per-band gains
2. Temporal EMA smoothing (if enabled)
3. RMS preservation (if enabled)  
4. Residual blend with original
5. Global gain application
6. Safety clamping (if enabled)

## Requirements

- ComfyUI
- Kijai's ComfyUI-WanVideoWrapper
- HuMo model weights

No additional Python dependencies needed.

## Troubleshooting

**"No visible effect":**
- Increase alpha_mix toward 1.0
- Increase global_gain
- Increase individual band gains
- Try more extreme band ratios

**"Black output":**
- Don't set bands 3 or 4 to 0.0 (model needs them)
- Reduce extreme gain values
- Enable preserve_rms
- Enable clamp_std

**"Flickering motion":**
- Increase ema_beta for more smoothing
- Try preserve_rms = True

## Credits

Developed with assistance from ChatGPT while experimenting with audio embedding manipulation in HuMo.

## License

MIT
