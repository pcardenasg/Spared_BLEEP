import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import CLIPModel
from dataset import BLEEP_Dataset
from torch.utils.data import DataLoader

import os
import numpy as np

from utils import *

# Using spare library
from spared.datasets import get_dataset
from spared.metrics import get_metrics

def build_loaders_inference():
    print("Building loaders")
    
    # Get dataset from the values defined in args
    dataset = get_dataset(args.dataset)

    # Declare train and test datasets
    train_split = dataset.adata[dataset.adata.obs['split']=='train']
    val_split = dataset.adata[dataset.adata.obs['split']=='val']
    test_split = dataset.adata[dataset.adata.obs['split']=='test']

    test_available = False if test_split.shape[0] == 0 else True

    transform = torchvision.transforms.Normalize(mean=mean, std=std)

    train_dataset = BLEEP_Dataset(train_split, args.prediction_layer, transform)
    val_dataset = BLEEP_Dataset(val_split, args.prediction_layer, transform)
    test_loader = None
    if test_available:
        test_dataset = BLEEP_Dataset(test_split, args.prediction_layer, transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True)

    # Set up distributed sampler
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=args.shuffle, pin_memory=True, drop_last=True)

    print("Finished building loaders")
    num_genes = train_split.X.shape[-1]

    return train_loader, val_loader, test_loader, num_genes

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
            # FIXME: revisar resultados de predicción sobre c_t_log1p sin enviar la máscara "mask"
            spot_expressions.append(batch["reduced_expression"])
            test_masks.append(batch["mask"].cuda())
    return torch.cat(spot_embeddings), torch.cat(spot_expressions).cuda(), torch.cat(test_masks)


#train_sizex256, test_sizex256
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

def inference_bleep(model_path, method = "average"):

    # Load data
    train_loader, val_loader, test_loader, num_genes = build_loaders_inference()

    # Build model
    model = CLIPModel(args, num_genes, train_loader = train_loader).cuda()

    # Load trained weights
    state_dict = torch.load(model_path)
    state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)

    # Set model to eval mode
    model.eval()
    print("Finished loading model")

    # Build image embeddings for test data using trained model
    img_embeddings_test = get_image_embeddings(model, test_loader)

    # Build spot embeddings for train and test data using the trained model
    spot_embeddings_test, spot_expressions_test, test_masks = get_spot_embeddings(model, test_loader)
    spot_embeddings_train, spot_expressions_train, _ = get_spot_embeddings(model, train_loader)

    # Define query
    # image_query = img_embeddings_test # shape: [4970, 256] -> spots query, embed_dim
    # expression_gt = spot_expressions_test #shape: [4970, 128] -> spots query, genes
    # spot_key = spot_embeddings_train #shape: [4910, 256] -> spots train, embed_dim
    # expression_key = spot_expressions_train #shape: [4910, 128] spots train, genes


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

if __name__ == "__main__":
    
    os.environ['USE_PYGEOS'] = '0' # To supress a warning from geopandas

    parser = get_main_parser()
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    mean = [0.5476, 0.5218, 0.6881]
    std  = [0.2461, 0.2101, 0.1649]
    
    #ckpt_path = "Results_BLEEP/villacampa_lung_organoid/2024-01-11-01-48-56/epoch=329-step=990.ckpt"
    ckpt_path = args.ckpt_path
    true, pred, test_masks = inference_bleep(model_path = ckpt_path)

    true = np.asarray(true.cpu())
    pred = np.asarray(pred.cpu())

    print(pred.shape)
    print(true.shape)
    print(np.max(pred))
    print(np.max(true))
    print(np.min(pred))
    print(np.min(true))

    #genewise correlation
    corr = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        corr[i] = np.corrcoef(pred[i,:], true[i,:],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr = corr[~np.isnan(corr)]
    print("Mean correlation across cells: ", np.mean(corr))

    corr = np.zeros(pred.shape[1])
    for i in range(pred.shape[1]):
        corr[i] = np.corrcoef(pred[:,i], true[:pred.shape[0],i],)[0,1] #corrcoef returns a matrix
    #remove nan
    corr = corr[~np.isnan(corr)]
    print("number of non-zero genes: ", corr.shape[0])
    print("mean correlation: ", np.mean(corr))
    print("max correlation: ", np.max(corr))
    print("number of genes with correlation > 0.3: ", np.sum(corr > 0.3))

    full_metrics = get_metrics(true, pred, test_masks.cpu())
    print("Full metrics: ", full_metrics)
