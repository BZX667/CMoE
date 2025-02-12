# CMoE
This repository contains a implementation of our "Improving Alignment and Uniformity of Expert Representation with Contrastive Learning for Mixture-of-Experts model" 

## Environment Setup
1.tensorflow=1.15  
2.python=3.6.8

## Guideline

### models
```EAR.py``` Expert Agreement Regularization  
```EHP.py``` Expert Homogeneity Penalty  
```data_augment.py``` Data augment methods  
```project_head.py``` Projection head function  
```model``` MoE based model including MMoE, PLE, etc.  

### Example to run the codes
bash MMOE.sh \
bash PLE.sh

### Application
The paper "Improving Alignment and Uniformity of Expert Representation with Contrastive Learning for Mixture of Experts Model" is applied to the Deepseek MoE model (scheduled for release in August 2024). By introducing orthogonal constraints between the hidden states of different experts, it alleviates issues such as imbalanced expert utilization, representation degradation, and representation collapse.
