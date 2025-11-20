# CRCL and Bi-CRCL for Class-incremental Medical Image Analysis with Pre-trained Foundation Models
This repository contains the official implementation of the paper:

MICCAI 2025: Conservative-Radical Complementary Learning for Class-incremental Medical Image Analysis with Pre-trained Foundation Models
Under Review: Bi-CRCL: Bidirectional Conservative-Radical Complementary Learning with Pre-trained Foundation Models for Class-incremental Medical Image Analysis

## Abstract
Class-incremental learning (CIL) in medical image-guided diagnosis requires models to retain diagnostic expertise on previously learned disease categories while continually adapting to newly emerging ones, which is a key step toward scalable clinical deployment. This problem is particularly challenging due to heterogeneous clinical data and privacy constraints that preclude memory replay. Although pretrained foundation models (PFMs) have revolutionized general-domain CIL through transferable and expressive representations, their potential in medical imaging remains underexplored, where domain-specific adaptation is essential yet challenging due to anatomical complexity and inter-institutional heterogeneity. To bridge this gap, we first conduct a systematic benchmark of recent PFM-based CIL methods in the medical domain and further propose Bidirectional Conservative-Radical Complementary Learning (Bi-CRCL), a dual-learner framework inspired by the brain’s complementary learning systems. Bi-CRCL comprises two synergistic PFM-based learners: (i) a \textbf{conservative learner} (neocortex-like) that preserves accumulated diagnostic knowledge through stability-oriented updates, and (ii) a \textbf{radical learner} (hippocampus-like) that rapidly acquires new categories via plasticity-oriented adaptation. Specifically, the dual-learner cross-classification alignment mechanism harmonizes their complementary strengths, reconciling inter-task decision boundaries to mitigate catastrophic forgetting. At the core of Bi-CRCL lies a bidirectional design mirroring hippocampus–neocortex interaction: prior to each new task, the radical learner is initialized with the conservative learner’s consolidated weights (forward transfer); after adaptation, the radical learner’s updates are progressively integrated back into the conservative learner via exponential moving average (backward consolidation). This cyclic exchange enables the continual integration of new knowledge while preserving prior expertise. During task-agnostic inference, Bi-CRCL adaptively fuses outputs from both learners to achieve robust final predictions. Comprehensive experiments on five medical imaging datasets validate Bi-CRCL’s effectiveness over state-of-the-art methods. Further evaluations across different PFMs, severe cross-dataset distribution shifts, varying task granularities, and reversed task orders confirm its robustness, scalability, and strong generalization capacity. 

## Requirements
- Python >= 3.8  
- PyTorch >= 1.4 
- tqdm == 4.65.0
- timm == 0.6.5

## Datasets
We follow [ACL](https://github.com/GiantJun/CL_Pytorch/tree/main) to use the same data index_list for training.

## Running scripts
### 1. Standard Usage
```bash
$ python main.py -d skin8
- for -d choose from 'medmnist', 'colon', 'blood', 'covid'
```
### 2. Select the Model (CRCL or BiCRCL)
- To switch between the CRCL and BiCRCL learners, edit the import line in trainer.py: from CRCL/BiCRCL import Learner


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
