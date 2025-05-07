# Predicting Poisson’s Ratio:A Study of Semisupervised Anomaly Detection and Supervised Approaches
The code given here implements the models discussed in the following paper. Contains novel application of semi-supervised anomaly detection algorithms for a Material Science Problem
[scheme](model.png)
[Link to the Paper](https://pubs.acs.org/doi/10.1021/acsomega.3c08861)
## How to cite
Please cite using  

```
@article{doi:10.1021/acsomega.3c08861,
author = {Hammad, Raheel and Mondal, Sownyak},
title = {Predicting Poisson’s Ratio: A Study of Semisupervised Anomaly Detection and Supervised Approaches},
journal = {ACS Omega},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1021/acsomega.3c08861},
URL = {https://doi.org/10.1021/acsomega.3c08861},
eprint = {https://doi.org/10.1021/acsomega.3c08861}
}
```
## Abstract
Auxetics are a rare class of materials that exhibit a negative Poisson’s ratio. The existence of these auxetic materials is rare but has a large number of applications in the design of exotic materials. We build a complete machine learning framework to detect Auxetic materials as well as Poisson’s ratio of non-auxetic materials. A semisupervised anomaly detection model is presented, which is capable of separating out the auxetics materials (treated as an anomaly) from an unknown database with an average precision of 0.64. Another regression model (supervised) is also created to predict the Poisson’s ratio of non-auxetic materials with an R2 of 0.82. Additionally, this regression model helps us to find the optimal features for the anomaly detection model. This methodology can be generalized and used to discover materials with rare physical properties.
