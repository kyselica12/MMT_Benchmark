from transforms import *


config = {
    "model": "FURFARO",
    "path": "~/work/mmt",
    "dataset_dir": "./RoBo6", # Path to the dataset 
    "input_size": 10_000,
    "n_classes": 6,
    "in_channels": 1,
    "epochs": 50,
    "lr": 0.001,
    "batch_size": 32,
    "channels": ["mag"],
    "transform": lambda x: x,
    "trainer_args": {},
}

allworth_config = config.copy()
allworth_config.update({
    "model": "ALLWORTH",
    "input_size": 1_200,
    "channels": ["mag", "phase"],
    "in_channels": 2,
    "batch_size": 256,
    "transform": allworth_transform
})

furfaro_config = config.copy()
furfaro_config.update({
    "model": "FURFARO",
    "transform": furfaro_transform,
    "input_size": 500,
    "batch_size": 128
})

astroconformer_config = config.copy()
astroconformer_config.update({
    "model": "ASTROCONFORMER",
    "transform": astroconformer_transform,
    "trainer_args":{
        "precision": 16,
        "devices": [0]
    },
})
    
yao_config = config.copy()
yao_config.update({
   "model": "YAO",
   "input_size": 200,
   "transform": yao_transform
})

resnet_config = config.copy()
resnet_config.update({
    "model": "RESNET",
})

config_dict = {
    "ALLWORTH": allworth_config,
    "FURFARO": furfaro_config,
    "ASTROCONFORMER": astroconformer_config,
    "YAO": yao_config,
    "RESNET": resnet_config
}

