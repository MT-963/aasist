# AASIST2 and Speaker-Aware Enhancements

This document explains the enhancements made to the original AASIST architecture for spoofing detection. These enhancements are based on the AASIST2 and Speaker-Aware papers and aim to improve performance, especially on short utterances.

## Implemented Enhancements

### 1. Feature Encoder: Res2Net Blocks with Squeeze-and-Excitation

The original ResNet blocks in AASIST have been replaced with Res2Net blocks followed by squeeze-and-excitation layers. This enables multi-scale feature extraction and improves performance, especially on short segments.

- **Implementation:** `Res2NetBlock` class in `models/AASIST.py`
- **Configuration:**
  - `res2net_width`: Width of the Res2Net block (default: 14)
  - `res2net_scale`: Scale factor for the Res2Net block (default: 8)

### 2. Loss & Training Strategy: AM-Softmax + DCS + ALMFT

#### 2.1 AM-Softmax Loss

Replaced the standard cross-entropy loss with AM-Softmax (Additive Margin Softmax) loss, which adds a margin to the target logit.

- **Implementation:** `AMSoftmaxLoss` class in `utils.py`
- **Configuration:**
  - `loss`: Set to "AM_Softmax" to use this loss
  - `am_softmax_scale`: Scale factor (s) (default: 15.0)
  - `adaptive_margin`: Whether to use adaptive margin based on duration (default: true)

#### 2.2 Dynamic Chunk Size (DCS)

During training, randomly truncates/pads utterances to a length between 1-6 seconds (16000-96000 samples at 16kHz), which helps the model learn to handle variable-length inputs better.

- **Implementation:** `dynamic_chunk_size` function in `data_utils.py`
- **Configuration:**
  - `dynamic_chunk.enabled`: Whether to enable dynamic chunk size (default: true)
  - `dynamic_chunk.min_samples`: Minimum number of samples (default: 16000)
  - `dynamic_chunk.max_samples`: Maximum number of samples (default: 96000)

#### 2.3 Adaptive Large Margin Fine-Tuning (ALMFT)

The margin in AM-Softmax is computed dynamically based on utterance duration: m = A * duration + B.

- **Implementation:** Incorporated in `AMSoftmaxLoss` class in `utils.py`
- **Configuration:**
  - `margin_a`: Slope (A) for margin calculation (default: 0.06 or 3/50)
  - `margin_b`: Intercept (B) for margin calculation (default: 0.14 or 7/50)

### 3. Speaker-Aware Conditioning

The model can now incorporate speaker identity embeddings to condition the detection decision, making it more robust against sophisticated spoofing attacks that mimic specific speakers.

- **Implementation:** `SpeakerConditioningModule` class in `models/AASIST.py`
- **Configuration:**
  - `speaker_conditioning`: Whether to enable speaker conditioning (default: true)
  - `spk_emb_dim`: Dimension of speaker embedding vectors (default: 256)
  - `conditioning_level`: Where to apply conditioning - "frame" or "utterance" (default: "frame")
  - `use_attention`: Whether to use attention mechanism for conditioning (default: true)

## Usage

### Configuration

Use the `config/AASIST2.conf` configuration file as a starting point:

```bash
py main.py --config config/AASIST2.conf
```

### Using Speaker Embeddings

To use speaker embeddings:

1. Extract speaker embeddings from an enrollment utterance using a speaker verification system
2. Pass the speaker embeddings to the model during forward pass:

```python
# Example code for using speaker embeddings
speaker_embedding = extract_speaker_embedding(enrollment_utterance)  # Your speaker embedding extractor
_, output = model(audio_input, speaker_embedding=speaker_embedding)
```

## Performance Improvements

According to the papers, these enhancements provide:
- Better performance on short utterances
- Lower Equal Error Rate (EER) by ↓25.1% 
- Lower minimum t-DCF by ↓11.6% on ASVspoof2019

## References

1. [AASIST2: Improving Short Utterance Anti-Spoofing with AASIST2](https://arxiv.org/abs/2309.08279)
2. [Speaker-Aware Anti-spoofing](https://www.isca-archive.org/interspeech_2023/liu23o_interspeech.html) 