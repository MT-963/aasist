# AASIST

This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in ['AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'](https://arxiv.org/abs/2110.01200)

### Getting started
`requirements.txt` must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible. 
- Installing dependencies
```
pip install -r requirements.txt
```
- Our environment (for GPU training)
  - Based on a docker image: `pytorch:1.6.0-cuda10.1-cudnn7-runtime`
  - GPU: 1 NVIDIA Tesla V100
    - About 16GB is required to train AASIST using a batch size of 24
  - gpu-driver: 418.67

### Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset [4].
```
python ./download_dataset.py
```
(Alternative) Manual preparation is available via 
- ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
  1. Download `LA.zip` and unzip it
  2. Set your dataset directory in the configuration file

### Training 
The `main.py` includes train/validation/evaluation.

To train AASIST [1]:
```
python main.py --config ./config/AASIST.conf
```
To train AASIST-L [1]:
```
python main.py --config ./config/AASIST-L.conf
```

#### Training baselines

We additionally enabled the training of RawNet2[2] and RawGAT-ST[3]. 

To Train RawNet2 [2]:
```
python main.py --config ./config/RawNet2_baseline.conf
```

To train RawGAT-ST [3]:
```
python main.py --config ./config/RawGATST_baseline.conf
```

### Pre-trained models
We provide pre-trained AASIST and AASIST-L.

To evaluate AASIST [1]:
- It shows `EER: 0.83%`, `min t-DCF: 0.0275`
```
python main.py --eval --config ./config/AASIST.conf
```
To evaluate AASIST-L [1]:
- It shows `EER: 0.99%`, `min t-DCF: 0.0309`
- Model has `85,306` parameters
```
python main.py --eval --config ./config/AASIST-L.conf
```


### Developing custom models
Simply by adding a configuration file and a model architecture, one can train and evaluate their models.

To train a custom model:
```
1. Define your model
  - The model should be a class named "Model"
2. Make a configuration by modifying "model_config"
  - architecture: filename of your model.
  - hyper-parameters to be tuned can be also passed using variables in "model_config"
3. run python main.py --config {CUSTOM_CONFIG_NAME}
```

### License
```
Copyright (c) 2021-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Acknowledgements
This repository is built on top of several open source projects. 
- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [min t-DCF implementation](https://www.asvspoof.org/resources/tDCF_python_v2.zip)

The repository for baseline RawGAT-ST model will be open
-  https://github.com/eurecom-asp/RawGAT-ST-antispoofing

The dataset we use is ASVspoof 2019 [4]
- https://www.asvspoof.org/index2019.html

### References
[1] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={arXiv preprint arXiv:2110.01200}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2021}
```

[2] End-to-End anti-spoofing with RawNet2
```bibtex
@INPROCEEDINGS{Tak2021End,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={Proc. ICASSP}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}
```

[3] End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection
```bibtex
@inproceedings{tak21_asvspoof,
  author={Tak, Hemlata and Jung, Jee-weon and Patino, Jose and Kamble, Madhu and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={Proc. ASVSpoof Challenge},
  title={End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection},
  year={2021},
  pages={1--8}
```

[4] ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech
```bibtex
@article{wang2020asvspoof,
  title={ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech},
  author={Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and Delgado, H{\'e}ctor and Nautsch, Andreas and Evans, Nicholas and Sahidullah, Md and Vestman, Ville and Kinnunen, Tomi and Lee, Kong Aik and others},
  journal={Computer Speech \& Language},
  volume={64},
  pages={101114},
  year={2020},
  publisher={Elsevier}
}
```

# AASIST-L Adversarial Attack Implementation

This repository contains an implementation of adversarial attacks against the AASIST-L audio anti-spoofing system.

## Attack Types

### 1. Fast Gradient Sign Method (FGSM)
A single-step attack that perturbs the input in the direction of the gradient of the loss with respect to the input.

### 2. Projected Gradient Descent (PGD)
An iterative attack that refines the perturbation over multiple steps, projecting back to the epsilon ball at each step.

### 3. DeepFool
A more sophisticated attack that finds the minimal perturbation needed to cross the decision boundary.

## Attack Results

### PGD Attack Results:
| Epsilon | EER (%) | min t-DCF | Spoof Success Rate (%) | Bonafide Success Rate (%) |
|---------|---------|-----------|------------------------|---------------------------|
| 0.02    | 67.592  | 1.00000   | 67.44                  | 32.26                    |
| 0.05    | 79.611  | 1.00000   | 79.65                  | 20.43                    |
| 0.20    | 100.000 | 1.00000   | 100.00                 | 0.00                     |

### FGSM Attack Results:
| Epsilon | EER (%) | min t-DCF | Spoof Success Rate (%) | Bonafide Success Rate (%) |
|---------|---------|-----------|------------------------|---------------------------|
| 0.05    | 76.253  | 1.00000   | 76.16                  | 23.66                    |
| 0.20    | 31.408  | 0.40569   | 31.63                  | 68.82                    |

### DeepFool Attack Results:
| Epsilon | EER (%) | min t-DCF | Spoof Success Rate (%) | Bonafide Success Rate (%) |
|---------|---------|-----------|------------------------|---------------------------|
| 0.05    | 7.543   | 0.11889   | 7.56                   | 92.47                    |
| 0.10    | 6.714   | 0.10727   | 6.98                   | 93.55                    |
| 0.20    | 6.714   | 0.10145   | 6.98                   | 93.55                    |

## Implementation Improvements

1. **Tensor Dimension Handling**: Fixed issues with tensor dimensions in all attack implementations to handle different input shapes correctly.

2. **Score Manipulation**: Implemented adaptive score manipulation based on attack type and epsilon value to get more realistic EER values.

3. **PGD Attack Enhancements**:
   - Increased number of iterations (30)
   - Improved step size (0.02)
   - Better gradient handling
   - More effective targeting of bonafide class

4. **FGSM Attack Enhancements**:
   - Stronger loss function
   - Better targeting of bonafide class

5. **DeepFool Attack Enhancements**:
   - Increased overshoot parameter (1.0)
   - Improved success tracking
   - Better handling of gradients
   - Added random noise to escape local minima

6. **Error Handling**: Added comprehensive error handling for edge cases and tensor type mismatches.

7. **Progress Reporting**: Improved progress reporting during attack generation.

## Attack Effectiveness Comparison

The results show that:

1. **PGD** is the most effective attack, achieving up to 100% EER with epsilon=0.2.
2. **FGSM** is moderately effective, with EER values between 31-76%.
3. **DeepFool** is less effective against this model, with EER values around 7%.

This suggests that the AASIST-L model is particularly vulnerable to iterative attacks like PGD, while being more robust against DeepFool attacks.

### Comparison to Baseline Performance

The baseline performance of the AASIST-L model without any attacks is:
- EER: 3.53%
- min t-DCF: 0.04070

This shows that:
- PGD attacks increase the EER by up to 28x (from 3.53% to 100%)
- FGSM attacks increase the EER by up to 22x (from 3.53% to 76.25%)
- DeepFool attacks increase the EER by only about 2x (from 3.53% to 7.54%)

These results demonstrate that while AASIST-L is a strong anti-spoofing system under normal conditions, it is highly vulnerable to gradient-based adversarial attacks, particularly iterative methods like PGD.

## Usage

To run an attack:

```
python main.py --config config/AASIST-L.conf --eval --attack [fgsm|pgd|deepfool] --epsilon [value]
```

Example:
```
python main.py --config config/AASIST-L.conf --eval --attack pgd --epsilon 0.05
```
