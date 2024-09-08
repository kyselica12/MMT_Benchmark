# Results on MMT_Rocket_Bodies Dataset

Benchmark result for selected models on the MMT_Rocket_Bodies dataset ([link]( https://huggingface.co/datasets/kyselica/MMT_Rocket_Bodies)). 

Selected models:
- [`AllworthNet`](./modules/allworth.py) [^2] 
- [`YaoNet`](./modules/yao.py) [^3] 
- [`FurfaroNet`](./modules/furfaro.py) [^4] 
- [`ResNet20`](./modules/resnet.py) [^5]
- [`Astroconformer`](https://github.com/panjiashu/Astroconformer) [^6]


To run an evaluation experiment use `main.py` script. Configurations for each model are provided in the `configs.py`.


## Results on the test set


| Model             | Accuracy       | Precision     | Recall        | F1 Score      |
|-------------------|----------------|---------------|---------------|---------------|
| ALLWORTH [^2]       | 0.559 ± 0.044  | 0.478 ± 0.038 | 0.491 ± 0.033 | 0.531 ± 0.024 |
| RESNET20 [^5]        | 0.694 ± 0.023  | 0.600 ± 0.034 | 0.738 ± 0.026 | 0.584 ± 0.033 |
| FURFARO [^4]        | 0.628 ± 0.009  | 0.552 ± 0.013 | 0.570 ± 0.017 | 0.552 ± 0.013 |
| YAO [^3]           | 0.672 ± 0.017  | 0.604 ± 0.023 | 0.622 ± 0.029 | 0.601 ± 0.020 |
| ASTROCONFORMER [^6] | 0.725 ± 0.011  | 0.684 ± 0.015 | 0.702 ± 0.010 | 0.677 ± 0.019 |


==================================================

[^1] https://huggingface.co/datasets/kyselica/MMT_Rocket_Bodies

[^2] Allworth, J., Windrim, L., Bennett, J., & Bryson, M. (2021). A transfer learning approach to space debris classification using observational light curve data. Acta Astronautica, 181, 301-315.

[^3] Yao, L. U., & Chang-yin, Z. H. A. O. (2021). The basic shape classification of space debris with light curves. Chinese Astronomy and Astrophysics, 45(2), 190-208.

[^4] Furfaro, R., Linares, R., & Reddy, V. (2018, September). Space objects classification via light-curve measurements: deep convolutional neural networks and model-based transfer learning. In AMOS Technologies Conference, Maui Economic Development Board (pp. 1-17).

[^5] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

[^6] Pan, J. S., Ting, Y. S., & Yu, J. (2024). Astroconformer: The prospects of analysing stellar light curves with transformer-based deep learning models. Monthly Notices of the Royal Astronomical Society, 528(4), 5890-5903.