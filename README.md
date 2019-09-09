# CASTNet
Opioid overdose is a growing public health crisis in the United States. This crisis, recognized as "opioid epidemic," has widespread societal consequences including the degradation of health, and the increase in crime rates and family problems. To improve the overdose surveillance and to identify the areas in need of prevention effort, in this work, we focus on forecasting opioid overdose using real-time crime dynamics. Previous work identified various types of links between opioid use and criminal activities, such as financial motives and common causes. Motivated by these observations, we propose a novel spatio-temporal predictive model for opioid overdose forecasting by leveraging the spatio-temporal patterns of crime incidents. Our proposed model incorporates multi-head attentional networks to learn different representation subspaces of features. Such deep learning architecture, called "community-attentive" networks, allows the prediction for a given location to be optimized by a mixture of groups (i.e., communities) of regions. In addition, our proposed model allows for interpreting what features, from what communities, have more contributions to predicting local incidents as well as how these communities are captured through forecasting. Our results on two real-world overdose datasets indicate that our model achieves superior forecasting performance and provides meaningful interpretations in terms of spatio-temporal relationships between the dynamics of crime and that of opioid overdose.

![CASTNet_architecture](https://user-images.githubusercontent.com/3327205/64565550-ea64e700-d34b-11e9-8c79-591b363d2ba6.png)

This repository contains Tensorflow implementation of [CASTNet](https://arxiv.org/abs/1905.04714) presented in our ECML-PKDD 2019 paper:

Ali Mert Ertugrul, Yu-Ru Lin, Tugba Taskaya-Temizel. CASTNet: Community-Attentive Spatio-Temporal Networks for Opioid Overdose Forecasting. In ECML-PKDD, 2019.

## Citation
If you use any of the resources provided on this page, please cite the following paper:

```
@inproceedings{ertugrul2019CASTNet,
  title = {CASTNet: Community-Attentive Spatio-Temporal Networks for Opioid Overdose Forecasting},
  author = {Ertugrul, Ali Mert and Lin, Yu-Ru and Taskaya-Temizel, Tugba},
  booktitle = {Proceedings of Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2019)},
  year = {2019}
}
```
