# CRCL: Class-incremental Medical Image Analysis with Pre-trained Foundation Models
This repository contains the official implementation of the paper:

MICCAI 2025: Conservative-Radical Complementary Learning for Class-incremental Medical Image Analysis with Pre-trained Foundation Models

## Abstract 
Class-incremental learning (CIL) in medical image-guided diagnosis requires models to retain diagnostic expertise on historical disease classes while adapting to newly emerging categories—a critical challenge for scalable clinical deployment. While pretrained foundation models (PFMs) have revolutionized CIL in the general domain by enabling generalized feature transfer, their potential remains underexplored in medical imaging, where domain-specific adaptations are critical yet challenging due to anatomical complexity and data heterogeneity. To address this gap, we first benchmark recent PFM-based CIL methods in the medical domain and further propose Conservative-Radical Complementary Learning (CRCL), a novel framework inspired by the complementary learning systems in the human brain. CRCL integrates two specialized learners built upon PFMs: (i) a neocortex-like conservative learner, which safeguards accumulated diagnostic knowledge through stability-oriented parameter updates, and (ii) a hippocampus-like radical learner, which rapidly adapts to new classes via dynamic and taskspecific plasticity-oriented optimization. Specifically, dual-learner feature and cross-classification alignment mechanisms harmonize their complementary strengths, reconciling inter-task decision boundaries to mitigate catastrophic forgetting. To ensure long-term knowledge retention while enabling adaptation, a consolidation process progressively transfers learned representations from the radical to the conservative learner. During task-agnostic inference, CRCL integrates outputs from both learners for robust final predictions. Comprehensive experiments on four medical imaging datasets show CRCL’s superiority over state-of-the-art methods.

## Requirements
- Python >= 3.8  
- PyTorch >= 1.4 
- tqdm == 4.65.0
- timm == 0.6.5

## Datasets
We follow [RanPAC](https://github.com/McDonnell-Research-Lab/RanPAC/tree/main), [ACL](https://github.com/GiantJun/CL_Pytorch/tree/main) settings to use the same data index_list for training.

## Running scripts
```bash
$ python main.py -d skin8
- for -d choose from 'medmnist', 'colon', 'blood', 'covid'
```

## Citation 
If you find our work useful in your research, please cite:
```bibtex
@inproceedings{wu2025crcl,
  title     = {Conservative-Radical Complementary Learning for Class-incremental Medical Image Analysis with Pre-trained Foundation Models},
  author    = {Wu*, Xinyao and Xu*, Zhe and Lu, Donghuan and Sun, Jinghan and Liu, Hong and Shakil, Sadia and Ma, Jiawei and Zheng, Yefeng and Tong, Raymond},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2025}
}
```
