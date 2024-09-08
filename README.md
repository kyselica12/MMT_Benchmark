# Results on MMT_Rocket_Bodies Dataset

Benchmark result for selected models on the MMT_Rocket_Bodies dataset [^1]. Results for individual models are obtained using the `main.py` script.

Selected models:
- [`AllworthNet`](./modules/allworth.py) [^2] 
- [`YaoNet`](./modules/yao.py) [^3] 
- [`FurfaroNet`](./modules/furfaro.py) [^4] 
- [`ResNet20`](./modules/resnet.py) [^5]
- [`Astroconformer`](https://github.com/panjiashu/Astroconformer) [^6]

Configurations for each model are provided in the `configs.py` script.


## Results on the test set

| Model          | Accuracy | Precision | Recall | F1 Score | 
|----------------|----------|-----------|--------|----------|
| AllworthNet    | 0.64     | 0.57      | 0.57   | 0.56     |
| YaoNet         | 0.68     | 0.63      | 0.60   | 0.61     |
| FurfaroNet     | 0.61     | 0.56      | 0.57   | 0.56     |
| ResNet20       | 0.74     | 0.74      | 0.70   | 0.71     |
| Astroconformer | 0.74     | 0.72      | 0.68   | 0.69     |



==================================================

[^1] https://huggingface.co/datasets/kyselica/MMT_Rocket_Bodies

[^2] Allworth, J., Windrim, L., Bennett, J., & Bryson, M. (2021). A transfer learning approach to space debris classification using observational light curve data. Acta Astronautica, 181, 301-315.

[^3] Yao, L. U., & Chang-yin, Z. H. A. O. (2021). The basic shape classification of space debris with light curves. Chinese Astronomy and Astrophysics, 45(2), 190-208.

[^4] Furfaro, R., Linares, R., & Reddy, V. (2018, September). Space objects classification via light-curve measurements: deep convolutional neural networks and model-based transfer learning. In AMOS Technologies Conference, Maui Economic Development Board (pp. 1-17).

[^5] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

[^6] Pan, J. S., Ting, Y. S., & Yu, J. (2024). Astroconformer: The prospects of analysing stellar light curves with transformer-based deep learning models. Monthly Notices of the Royal Astronomical Society, 528(4), 5890-5903.