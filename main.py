import wandb
import yaml
from datasets import load_dataset
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import torch

from dataset import MMTDataset
from modules.yao import YaoNet
from modules.furfaro import FurfaroNet
from lightning_module import LightningModule
from modules.allworth import AllworthNet
from modules.resnet import resnet20
from configs import config, yao_config, furfaro_config, allworth_config, astroconformer_config, resnet_config

from Astroconformer.Astroconformer.Train.utils import init_train
from Astroconformer.Astroconformer.utils import Container


def get_dataset(config):
    ds = load_dataset(config["dataset_dir"], data_files={"train": "train.csv", "test": "test.csv"})
    labels = sorted(list(set(ds["train"]["label"])))
    return ds, labels

def get_torch_datasets(config, dataset, labels):
    train_set = MMTDataset(config["dataset_dir"], dataset["train"], labels, transforms=config["transform"], channels=config["channels"])
    test_set = MMTDataset(config["dataset_dir"], dataset["test"], labels, transforms=config["transform"], channels=config["channels"])
    return train_set, test_set


def get_model(config):
    net, optimizer, scheduler = None, None, None
    match config["model"]:
        case "ALLWORTH":
            net = AllworthNet(config["in_channels"], config["input_size"], config["n_classes"])
        case "FURFARO":
            config = furfaro_config
            net = FurfaroNet(config["in_channels"], config["input_size"], config["n_classes"])

        case "ASTROCONFORMER":
            config = astroconformer_config

            args = Container(**yaml.safe_load(open('./Astroconformer/experiment_config.yaml', 'r')))
            args.load_dict(yaml.safe_load(open('./Astroconformer/model_config.yaml', 'r'))[args.model])
            args.load_dict(yaml.safe_load(open('./Astroconformer/default_config.yaml', 'r')))
            args.device = "cuda"
            args.input_shape = (1,config["input_size"])
            net, optimizer, scheduler, scaler = init_train(args)

        case "YAO":
            config = yao_config
            net = YaoNet(config["in_channels"], config["input_size"], config["n_classes"])
        
        case "RESNET":
            net = resnet20(config["n_classes"], config["in_channels"])
    
    return net, optimizer, scheduler
    

def train(net, optimizer, scheduler, train_set, test_set, config, logger=None, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    # net = resnet20(len(labels), IN_CHANNELS)
    module = LightningModule(net, 
                            n_classes=config["n_classes"],
                            lr=config["lr"],
                            optimizer=optimizer,
                            scheduler=scheduler,
                            experiment_name=config["model"])

    trainer = Trainer(default_root_dir='MMT_Logs', 
                        max_epochs=config["epochs"],
                        logger=logger,
                        **config["trainer_args"])

    print("Starting training...")
    trainer.fit(module, train_loader, test_loader)

    trainer.test(module, test_loader)

if __name__ == "__main__":
    N = 10
    dataset, labels = get_dataset(config)
    
    for config in [resnet_config, furfaro_config, allworth_config, yao_config, astroconformer_config]:
        train_set, test_set = get_torch_datasets(config, dataset, labels)

        # header for the results file
        with open(f"results/{config['model']}_test_resutls.csv", "w") as f:
            text = "time,acc,f1,prec,rec,"
            text += ','.join([f"{l}_f1,{l}_prec,{l}_rec"  for l in labels])
            print(text, file=f)

        for i in range(N):
            logger = None
            # if using wandb
            # logger = WandbLogger(project="MMT", name=config["model"]) 
            net, optimizer, scheduler = get_model(config)
            train(net, optimizer, scheduler, train_set, test_set, config, logger, seed=None)
            # wandb.finish()

