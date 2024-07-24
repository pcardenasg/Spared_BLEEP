import cv2
import pandas as pd
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, spatial_pos_path, barcode_path, reduced_mtx_path):
        #image_path is the path of an entire slice of visium h&e stained image (~2.5GB)
        
        #spatial_pos_csv
            #barcode name
            #detected tissue boolean
            #x spot index
            #y spot index
            #x spot position (px)
            #y spot position (px)
        
        #expression_mtx
            #feature x spot (alphabetical barcode order)
    
        #barcode_tsv
            #spot barcodes - alphabetical order

        self.whole_image = cv2.imread(image_path)
        self.spatial_pos_csv = pd.read_csv(spatial_pos_path, sep=",", header = None) 
        # self.expression_mtx = csr_matrix(sio.mmread(expression_mtx_path)).toarray()
        self.barcode_tsv = pd.read_csv(barcode_path, sep="\t", header = None) 
        self.reduced_matrix = np.load(reduced_mtx_path).T  #cell x features
        
        #print("Finished loading all files")

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        barcode = self.barcode_tsv.values[idx,0]
        v1 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,4].values[0]
        v2 = self.spatial_pos_csv.loc[self.spatial_pos_csv[0] == barcode,5].values[0]
        image = self.whole_image[(v1-112):(v1+112),(v2-112):(v2+112)]
        image = self.transform(image)
        #breakpoint()
        
        item['image'] = torch.tensor(image).permute(2, 0, 1).float() #color channel first, then XY
        item['reduced_expression'] = torch.tensor(self.reduced_matrix[idx,:]).float()  #cell x features (3467)
        item['barcode'] = barcode
        item['spatial_coords'] = [v1,v2]

        return item


    def __len__(self):
        return len(self.barcode_tsv)


class BLEEP_Dataset(torch.utils.data.Dataset):
    def __init__(self, adata, args, pred_layer, data_transforms = None):
        
        self.pred_layer = pred_layer
        self.image_transforms = data_transforms

        self.all_patches = adata.obsm['patches_scale_1.0']
        self.all_counts = torch.tensor(adata.layers[self.pred_layer])
        self.all_masks = torch.tensor(adata.layers['mask'])

        self.noisy_training = args.noisy_training
        if self.noisy_training:
            self.all_counts[~self.all_masks] = 0

    def transform(self, image):
        image = Image.fromarray(image)
        # Random flipping and rotations
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            image = TF.vflip(image)
        angle = random.choice([180, 90, 0, -90])
        image = TF.rotate(image, angle)
        return np.asarray(image)

    def __getitem__(self, idx):
        item = {}
        image = self.all_patches[idx] # image is type 'anndata._core.views.ArrayView'
        image = image.reshape((round(np.sqrt(image.shape[0]/3)), round(np.sqrt(image.shape[0]/3)), -1))
        image = self.transform(image)
        image = torch.tensor(image).permute(2, 0, 1)

        image = self.image_transforms(image/255.)

        item['image'] = image.float() 

        exp_matrix = self.all_counts[idx]
        item['reduced_expression'] = torch.tensor(exp_matrix).float()  #spots x genes

        item['mask'] = self.all_masks[idx]

        return item


    def __len__(self):
        return self.all_masks.shape[0]