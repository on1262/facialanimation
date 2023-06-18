# EXPRESSIVE SPEECH-DRIVEN FACIAL ANIMATION WITH CONTROLLABLE EMOTIONS

Source code for: [Expressive Speech-driven Facial Animation with controllable emotions](https://arxiv.org/abs/2301.02008)

**This repository is under construction now**

Update:

- initial update(Done, 6.18)
- refactor framework
- test reproducibility
- update 3rd packages
- add deployment scripts

## Deployment

Not avaliable now.

## Dataset

**CREMA-D**

dataset link: [https://www.kaggle.com/datasets/ejlok1/cremad](https://www.kaggle.com/datasets/ejlok1/cremad)

remove bad sample: 1007_ITH_NEU_XX, 1064_IEO_DIS_MD

**LRS2**

**RAVDESS**

**VOCASET**

## Experiments

**test loss in VOCASET**

Models  | Max loss(mm)| average loss(mm) |
--------- | --------| --------|
random init  | 3.87 | 2.31 |
faceformer | 3.33 | 1.97 |
voca | 3.41 | 1.94 |
tf_emo_4(mouth loss coeff=0.5) | 3.24 | 1.92 |
tf_emo_5(jaw loss coeff=0) | 3.22 | 1.92 |
tf_emo_8(few mask+LRS2) | 3.36 | 2.01
(conf A)tf_emo_2(mouth loss coeff=0) | 3.39 | 2.01 |
(conf B)tf_emo_6(add params mask) | 3.29 | 1.97 |
(conf C)tf_emo_3(reduce noise) | 3.32 | 1.99 |
(conf D)tf_emo_7(add params mask and introduces LRS2 dataset) | 3.34 | 1.99 |
(conf E)use only vocaset | 3.30 | 1.97 |
(conf F)tf_emo_10(disable transformer encoder) | 3.38 | 2.03 |