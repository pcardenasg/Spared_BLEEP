import numpy as np
import pandas as pd
import torchvision
import json
import squidpy as sq
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import os
import torch
import wandb
from tqdm import tqdm
import argparse
from spared.metrics import get_metrics
from anndata.experimental.pytorch import AnnLoader
from torch.utils.data import DataLoader
import plotly.express as px
from dataset import BLEEP_Dataset
import torch.nn.functional as F


# Auxiliary function to use booleans in parser
str2bool = lambda x: (str(x).lower() == 'true')
str2intlist = lambda x: [int(i) for i in x.split(',')]
str2floatlist = lambda x: [float(i) for i in x.split(',')]
str2h_list = lambda x: [str2intlist(i) for i in x.split('//')[1:]]

# Function to get global parser
def get_main_parser():
    parser = argparse.ArgumentParser(description='Code for expression prediction using contrastive learning implementation.')
    # Parameters #############################################################################################################################################################################
    -- parser.add_argument('--dataset',                    type=str,           default='10xgenomic_human_brain',   help='Dataset to use.')
    -- parser.add_argument('--prediction_layer',           type=str,           default='c_d_log1p',                help='The prediction layer from the dataset to use.')
    -- parser.add_argument('--noisy_training',             type=str2bool,      default=False,                      help='Whether or not to use noisy gene expression for training.')
    -- parser.add_argument('--max_steps',                  type=int,           default=1000,                       help='Number of steps to train de model.')
    -- parser.add_argument('--val_check_interval',         type=int,           default=10,                         help='Number of steps to do valid checks.')
    -- parser.add_argument('--batch_size',                 type=int,           default=256,                        help='The batch size to train model.')
    -- parser.add_argument('--shuffle',                    type=str2bool,      default=True,                       help='Whether or not to shuffle the data in dataloaders.')
    -- parser.add_argument('--lr',                         type=float,         default=1e-2,                       help='Learning rate to use.')
    -- parser.add_argument('--optimizer',                  type=str,           default='Adam',                     help='Optimizer to use in training. Options available at: https://pytorch.org/docs/stable/optim.html It will just modify main optimizers and not sota (they have fixed optimizers).')
    -- parser.add_argument('--momentum',                   type=float,         default=0.9,                        help='Momentum to use in the optimizer if it receives this parameter. If not, it is not used. It will just modify main optimizers and not sota (they have fixed optimizers).')
    -- parser.add_argument('--exp_name',                   type=str,           default='None',                     help='Name of the experiment to save in the results folder. "None" will assign a date coded name.')
    ##########################################################################################################################################################################################

    return parser

def get_image_embeddings(model, loader):
    model.eval()
    
    test_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            image_features = model.image_encoder(batch["image"].cuda())
            image_embeddings = model.image_projection(image_features)
            test_image_embeddings.append(image_embeddings)
    
    return torch.cat(test_image_embeddings)

def get_spot_embeddings(model, loader):
    model.eval()

    spot_embeddings = []
    spot_expressions = []
    test_masks = []
    with torch.no_grad():
        for batch in tqdm(loader):
            spot_embeddings.append(model.spot_projection(batch["reduced_expression"].cuda()))

            spot_expressions.append(batch["reduced_expression"])
            test_masks.append(batch["mask"].cuda())
    return torch.cat(spot_embeddings), torch.cat(spot_expressions).cuda(), torch.cat(test_masks)


#2265x256, 2277x256 -> train_sizex256, test_sizex256
def find_matches(spot_embeddings, query_embeddings, top_k=1):
    #find the closest matches 
    spot_embeddings = torch.tensor(spot_embeddings)
    query_embeddings = torch.tensor(query_embeddings)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
    spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
    dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265 -> test_size x train_size
    print(dot_similarity.shape)
    _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
    
    return indices

def inference_bleep(model, train_loader, test_loader, method = "average"):

    # Build image embeddings for test data using trained model
    img_embeddings_test = get_image_embeddings(model, test_loader)

    # Build spot embeddings for train and test data using the trained model
    spot_embeddings_test, spot_expressions_test, test_masks = get_spot_embeddings(model, test_loader)
    spot_embeddings_train, spot_expressions_train, _ = get_spot_embeddings(model, train_loader)

    # Define query
    # img_embeddings_test # shape: [4970, 256] -> spots query, embed_dim
    # spot_expressions_test #shape: [4970, 128] -> spots query, genes
    # spot_embeddings_train #shape: [4910, 256] -> spots train, embed_dim
    # spot_expressions_train #shape: [4910, 128] spots train, genes


    if method == "simple":
        indices = find_matches(spot_embeddings_train, img_embeddings_test, top_k=1)
        matched_spot_embeddings_pred = spot_embeddings_train[indices[:,0],:]
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        matched_spot_expression_pred = spot_expressions_train[indices[:,0],:]
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "average":
        print("finding matches, using average of top 50 expressions")
        indices = find_matches(spot_embeddings_train, img_embeddings_test, top_k=50)
        indices = torch.tensor(indices).cuda()
        matched_spot_embeddings_pred = torch.zeros((indices.shape[0], spot_embeddings_train.shape[1])).cuda()
        matched_spot_expression_pred = torch.zeros((indices.shape[0], spot_expressions_train.shape[1])).cuda()
        for i in range(indices.shape[0]):
            matched_spot_embeddings_pred[i,:] = torch.mean(spot_embeddings_train[indices[i,:],:], axis=0)
            matched_spot_expression_pred[i,:] = torch.mean(spot_expressions_train[indices[i,:],:], axis=0)
        
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape) # No se usa más adelante
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    if method == "weighted_average":
        print("finding matches, using weighted average of top 50 expressions")
        indices = find_matches(spot_embeddings_train, img_embeddings_test, top_k=50)
        matched_spot_embeddings_pred = np.zeros((indices.shape[0], spot_embeddings_train.shape[1]))
        matched_spot_expression_pred = np.zeros((indices.shape[0], spot_expressions_train.shape[1]))
        for i in range(indices.shape[0]):
            a = np.sum((spot_embeddings_train[indices[i,0],:] - img_embeddings_test[i,:])**2) #the smallest MSE
            weights = np.exp(-(np.sum((spot_embeddings_train[indices[i,:],:] - img_embeddings_test[i,:])**2, axis=1)-a+1))
            if i == 0:
                print("weights: ", weights)
            matched_spot_embeddings_pred[i,:] = np.average(spot_embeddings_train[indices[i,:],:], axis=0, weights=weights)
            matched_spot_expression_pred[i,:] = np.average(spot_expressions_train[indices[i,:],:], axis=0, weights=weights)
        
        print("matched spot embeddings pred shape: ", matched_spot_embeddings_pred.shape)
        print("matched spot expression pred shape: ", matched_spot_expression_pred.shape)

    true = spot_expressions_test
    pred = matched_spot_expression_pred

    return true, pred, test_masks

def get_predictions(adata, args, model, train_loader, transform, layer: str = 'c_d_log1p', batch_size: int = 128, use_cuda: bool = False)->None:
    """
    _summary_

    Args:
        adata (ad.AnnData): The AnnData object to process. Must be already filtered.
        model (_type_): BLEEP model with best weights already loaded.
        train_loader (torch.DataLoader): DataLoader with only the train set.
        transform (torchvision function): Visual transformations for images.
        layer (str, optional): _description_. Defaults to 'c_d_log1p'.
        batch_size (int, optional): _description_. Defaults to 128.
        use_cuda (bool, optional): _description_. Defaults to False.
    """

    if isinstance(adata, DataLoader):
        # Means 'adata' is actually only test_split's dataloader and the results will be used for testing purposes.
        true, pred, test_masks = inference_bleep(model, train_loader, adata)

    else:

        # Set the X of the adata to the layer casted to float32
        adata.X = adata.layers[layer].astype(np.float32)

        # Get complete dataloader
        adata_bleep = BLEEP_Dataset(adata, args, layer, transform)
        #dataloader = AnnLoader(adata_bleep, batch_size=batch_size, shuffle=False, use_cuda=use_cuda)
        dataloader = DataLoader(adata_bleep, batch_size=batch_size, shuffle=False)

        # Set model to eval mode
        model.eval()

        # Get complete predictions
        true, pred, test_masks = inference_bleep(model, train_loader, dataloader)

        # Define global variables
        glob_expression_pred = pred
        glob_ids = adata.obs['unique_id'].tolist() 
            
        # Put predictions in a single dataframe
        pred_matrix = glob_expression_pred.detach().cpu().numpy()
        pred_df = pd.DataFrame(pred_matrix, index=glob_ids, columns=adata.var_names)
        pred_df = pred_df.reindex(adata.obs.index)

        # Log predictions to wandb
        wandb_df = pred_df.reset_index(names='sample')
        wandb.log({'predictions': wandb.Table(dataframe=wandb_df)})
        
        # Add layer to adata
        adata.layers[f'predictions,{layer}'] = pred_df

    return adata, true, pred, test_masks

def get_slide_from_collection(collection, slide):
    """
    This function receives a slide name and returns an adata object of the specified slide based on the collection of slides
    in collection.

    Args: 
        collection (ad.AnnData): AnnData object with all the slides.
        slide (str): Name of the slide to get from the collection. Must be in the column 'slide_id' of the obs dataframe of the collection.

    Returns:
        ad.AnnData: An anndata object with the specified slide.
    """

    # Get the slide from the collection
    slide_adata = collection[collection.obs['slide_id'] == slide].copy()
    # Modify the uns dictionary to include only the information of the slide
    slide_adata.uns['spatial'] = {slide: collection.uns['spatial'][slide]}

    # Return the slide
    return slide_adata

def log_pred_image(adata, n_genes: int = 2, slides: dict = {}):
    
    ### 1. Get the selected genes

    # Get prediction and groundtruth layers from layers keys in the anndata
    pred_layer = [l for l in adata.layers.keys() if 'predictions' in l]
    pred_layer = pred_layer[0] if pred_layer else None
    gt_layer = [l.split(',')[1] for l in adata.layers.keys() if 'predictions' in l]
    gt_layer = gt_layer[0] if gt_layer else None
    # Be sure the prediction layer is present in dataset
    assert not (pred_layer is None), 'predictions layer not present in the adata'

    # Get partition names of current dataset
    partitions = list(adata.obs.split.unique())
    # Get dict of adatas separated by splits
    adatas_dict = {p: adata[adata.obs.split == p, :] for p in partitions}

    # Compute and ad detailed metrics for each split
    for p, curr_adata in adatas_dict.items():

        # Get detailed metrics from partition
        detailed_metrics = get_metrics(
            gt_mat = curr_adata.to_df(layer=gt_layer).values,
            pred_mat = curr_adata.to_df(layer=pred_layer).values,
            mask = curr_adata.to_df(layer='mask').values,
            detailed=True
        )

        # Add detalied metrics to global adata
        adata.var[f'pcc_{p}'] = detailed_metrics['detailed_PCC-Gene']
        adata.var[f'r2_{p}'] = detailed_metrics['detailed_R2-Gene']
        adata.var[f'mse_{p}'] = detailed_metrics['detailed_mse_gene']
        adata.var[f'mae_{p}'] = detailed_metrics['detailed_mae_gene']
        adata.var[f'avg_err_{p}'] = detailed_metrics['detailed_error_gene']
        
        # Define plotly plots
        pcc_fig = px.histogram(adata.var, x=f'pcc_{p}', marginal='rug', hover_data=adata.var.columns)
        r2_fig = px.histogram(adata.var, x=f'r2_{p}', marginal='rug', hover_data=adata.var.columns)
        mse_fig = px.histogram(adata.var, x=f'mse_{p}', marginal='rug', hover_data=adata.var.columns)
        mae_fig = px.histogram(adata.var, x=f'mae_{p}', marginal='rug', hover_data=adata.var.columns)
        avg_err_fig = px.histogram(adata.var, x=f'avg_err_{p}', marginal='rug', hover_data=adata.var.columns)
        err_df = pd.DataFrame(detailed_metrics['detailed_errors'], columns=[f'error_{p}'])
        err_fig = px.histogram(err_df, x=f'error_{p}')

        # Log plotly plot to wandb
        wandb.log({f'pcc_gene_{p}': wandb.Plotly(pcc_fig)})
        wandb.log({f'r2_gene_{p}': wandb.Plotly(r2_fig)})
        wandb.log({f'mse_gene_{p}': wandb.Plotly(mse_fig)})
        wandb.log({f'mae_gene_{p}': wandb.Plotly(mae_fig)})
        wandb.log({f'avg_err_gene_{p}': wandb.Plotly(avg_err_fig)})
        wandb.log({f'err_{p}': wandb.Plotly(err_fig)})
        
    
    # Get ordering split
    order_split = 'test' if 'test' in partitions else 'val'
    # Get selected genes based on the best pcc
    n_top = adata.var.nlargest(n_genes, columns=f'pcc_{order_split}').index.to_list()
    n_botom = adata.var.nsmallest(n_genes, columns=f'pcc_{order_split}').index.to_list()
    selected_genes = n_top + n_botom


    ### 2. Get the selected slides. NOTE: Only first slide is always selected in case slides is not specified by parameter.
    if slides == {}:
        slides = {p: list(adatas_dict[p].obs.slide_id.unique())[0] for p in partitions}
    
    def log_one_gene(gene, slides, gt_layer, pred_layer):
        
        # Get dict of individual slides adatas
        slides_adatas_dict = {p: get_slide_from_collection(adata, slides[p]) for p in slides.keys()}

        # Get min and max of the selected gene in the slides
        gene_min_pred = min([dat[:, gene].layers[pred_layer].min() for dat in slides_adatas_dict.values()])
        gene_max_pred = max([dat[:, gene].layers[pred_layer].max() for dat in slides_adatas_dict.values()])
        
        gene_min_gt = min([dat[:, gene].layers[gt_layer].min() for dat in slides_adatas_dict.values()])
        gene_max_gt = max([dat[:, gene].layers[gt_layer].max() for dat in slides_adatas_dict.values()])
        
        gene_min = min([gene_min_pred, gene_min_gt])
        gene_max = max([gene_max_pred, gene_max_gt])

        # Define color normalization
        norm = matplotlib.colors.Normalize(vmin=gene_min, vmax=gene_max)

        # Declare figure
        fig, ax = plt.subplots(nrows=len(slides), ncols=4, layout='constrained')
        fig.set_size_inches(13, 3 * len(slides))

        # Define order of rows in dict
        order_dict = {'train': 0, 'val': 1, 'test': 2}

        # Iterate over partitions
        for p in slides.keys():
            
            # Get current row
            row = order_dict[p]
            
            curr_img = slides_adatas_dict[p].uns['spatial'][slides[p]]['images']['lowres']
            ax[row,0].imshow(curr_img)
            ax[row,0].set_ylabel(f'{p}:\n{slides[p]}\nPCC-Gene={round(adata.var.loc[gene, f"pcc_{p}"],3)}', fontsize='large')
            ax[row,0].set_xticks([])
            ax[row,0].set_yticks([])

            # Plot gt and pred of gene in the specified slides
            sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=gt_layer, ax=ax[row,1], cmap='jet', norm=norm, colorbar=False, title='')
            sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=pred_layer, ax=ax[row,2], cmap='jet', norm=norm, colorbar=False, title='')
            sq.pl.spatial_scatter(slides_adatas_dict[p], color=[gene], layer=pred_layer, ax=ax[row,3], cmap='jet', colorbar=True, title='')
            
            # Set y labels
            ax[row,1].set_ylabel('')
            ax[row,2].set_ylabel('', fontsize='large')
            ax[row,3].set_ylabel('')


        # Format figure
        for axis in ax.flatten():
            axis.set_xlabel('')
            # Turn off all spines
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)
            axis.spines['left'].set_visible(False)

        # Refine figure appereance
        ax[0, 0].set_title(f'Original Image', fontsize='large')
        ax[0, 1].set_title(f'Groundtruth\nFixed Scale', fontsize='large')
        ax[0, 2].set_title(f'Prediction\nFixed Scale', fontsize='large')
        ax[0, 3].set_title(f'Prediction\nVariable Scale', fontsize='large')

        # Add fixed colorbar
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax[:3, 2], location='right', fraction=0.05, aspect=25*len(slides)+5)

        # Get ordering split
        order_split = 'test' if 'test' in slides.keys() else 'val'
        fig.suptitle(f'{gene}: PCC_{order_split}={round(adata.var.loc[gene, f"pcc_{order_split}"],3)}', fontsize=20)
        # Log plot 
        wandb.log({gene: fig})
        plt.close()

    for gene in selected_genes:
        log_one_gene(gene, slides, gt_layer, pred_layer)

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]