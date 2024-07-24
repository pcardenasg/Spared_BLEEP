import os

import torch
import torch.utils.data.distributed

from dataset import CLIPDataset, BLEEP_Dataset
from models import CLIPModel
from torch.utils.data import DataLoader

from utils import *
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from pytorch_lightning import seed_everything

# Using spare library
from spared.datasets import get_dataset
os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas

parser = get_main_parser()
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

def build_loaders(args, train_split, val_split, test_split, transform):

    test_available = False if test_split.shape[0] == 0 else True

    train_dataset = BLEEP_Dataset(train_split, args, args.prediction_layer, transform)
    val_dataset = BLEEP_Dataset(val_split, args, args.prediction_layer, transform)
    test_loader = None
    if test_available:
        test_dataset = BLEEP_Dataset(test_split, args, args.prediction_layer, transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # Set up distributed sampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True)

    print("Finished building loaders")
    num_genes = train_split.X.shape[-1]
    return train_loader, val_loader, test_loader, num_genes


def main():
    print("Starting...")
    args = parser.parse_args()

    # Seed everything and get cuda
    seed_everything(42, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare results path
    if args.exp_name == 'None':
        args.exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    results_path = os.path.join("Results_BLEEP", args.dataset, args.exp_name)
    os.makedirs(results_path, exist_ok= True)
    
    # Start wandb configs
    wandb_logger = WandbLogger(
    project='spared_bleep_sota',
    name=args.exp_name,
    log_model=False
    )

    # Get dataset from the values defined in args
    dataset = get_dataset(args.dataset)

    # Declare train and test datasets
    train_split = dataset.adata[dataset.adata.obs['split']=='train']
    val_split = dataset.adata[dataset.adata.obs['split']=='val']
    test_split = dataset.adata[dataset.adata.obs['split']=='test']

    # Prepare visual transformations
    mean = [0.5476, 0.5218, 0.6881]
    std  = [0.2461, 0.2101, 0.1649]
    transform = torchvision.transforms.Normalize(mean=mean, std=std)
    
    #load the data
    train_loader, val_loader, test_loader, num_genes = build_loaders(args, train_split, val_split, test_split, transform)

    #make the model
    '''
    Originally, BLEEP's repository allows the use of different image encoders. However, for practicality reasons we only
    adapted their default model (CLIP with resnet50 as image encoder) to run using Pytorch Lightning. If you wish to experiment 
    with other encoders, we recommend adapting the remaining models in models.py to the Pytorch Lightning format.
    '''
    model = CLIPModel(args = args, spot_embedding = num_genes, train_loader = train_loader).to(device)
    print("Image encoder is ResNet50")

    # Define checkpoint callback to save best model in validation
    checkpoint_callback = ModelCheckpoint(
                            dirpath=results_path,
                            monitor='val_loss', # Choose your validation metric
                            save_top_k=1, # Save only the best model
                            mode='min', # Choose "max" for higher values or "min" for lower values
                        )

    # Define the trainier and fit the model
    trainer = L.Trainer(
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.val_check_interval,
        check_val_every_n_epoch=None,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=wandb_logger
    )

    wandb_logger.experiment.config.update({'dataset':args.dataset, 'lr':args.lr, 'noisy_training':args.noisy_training})
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    print("Finished training")

    # Load the best model after training
    best_model_path = checkpoint_callback.best_model_path
    
    if test_loader is not None:
        trainer.test(model = model, dataloaders = test_loader, ckpt_path = best_model_path)

    else: 
        trainer.test(model = model, dataloaders = val_loader, ckpt_path = best_model_path)

    state_dict = torch.load(best_model_path)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Finished loading model with weights from {best_model_path}")

    # Get predictions for the entire dataset
    adata, _, _, _ = get_predictions(dataset.adata, args, model, train_loader, transform, layer = 'c_d_log1p', batch_size = args.batch_size, use_cuda = True)

    # Create and log prediction visualizations
    log_pred_image(adata)
    
    print("Test finished and metrics logged")

if __name__ == "__main__":
    main()
