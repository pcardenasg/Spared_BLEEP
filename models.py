import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, ProjectionHead, ImageEncoder_ViT, ImageEncoder_ViT_L, ImageEncoder_CLIP, ImageEncoder_resnet101, ImageEncoder_resnet152
from utils import *
import lightning as L
from spared.metrics import get_metrics

class CLIPModel(L.LightningModule):
    def __init__(
        self,
        args,
        spot_embedding, #num_genes
        train_loader,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding
        ):
        super().__init__()
        self.num_genes = spot_embedding
        self.image_encoder = ImageEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #corresponds to the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature
        self.args = args

        # Modifications for lightning:
        # Define loss criterion
        self.loss_meter_train = AvgMeter()
        self.loss_meter_valid = AvgMeter()
        self.best_loss_valid = np.inf
        self.best_step = 0
        self.train_loss = None
        self.valid_loss = None

        # Inference
        self.train_loader = train_loader
        self.test_image_embeddings = []
        self.test_spot_embeddings = []
        self.test_spot_expressions = []
        self.train_spot_embeddings = []
        self.train_spot_expressions = []
        self.test_masks = []

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def get_image_embeddings(self, batch):
        image_features = self.image_encoder(batch["image"])
        image_embeddings = self.image_projection(image_features)
        
        return image_embeddings

    def get_spot_embeddings(self, batch):
        self.test_spot_embeddings.append(self.spot_projection(batch["reduced_expression"]))
        self.test_spot_expressions.append(batch["reduced_expression"])

    def get_spot_embeddings_train(self):
        for batch in tqdm(self.train_loader):
            batch = {k: v.cuda() for k, v in batch.items() if k == "image" or k == "reduced_expression"}
            self.train_spot_embeddings.append(self.spot_projection(batch["reduced_expression"]))
            self.train_spot_expressions.append(batch["reduced_expression"])

        self.train_spot_embeddings = torch.cat(self.train_spot_embeddings)
        self.train_spot_expressions = torch.cat(self.train_spot_expressions)

    # train_sizex256, test_sizex256
    def find_matches(self, spot_embeddings, query_embeddings, top_k=1):
        #find the closest matches 
        spot_embeddings = torch.tensor(spot_embeddings)
        query_embeddings = torch.tensor(query_embeddings)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        spot_embeddings = F.normalize(spot_embeddings, p=2, dim=-1)
        dot_similarity = query_embeddings @ spot_embeddings.T   #2277x2265 -> test_size x train_size

        _, indices = torch.topk(dot_similarity.squeeze(0), k=top_k)
        
        return indices

    def training_step(self, batch):
        batch = {k: v.cuda() for k, v in batch.items()}
        # Get loss from the model
        loss = self.forward(batch)
        self.train_loss = loss.item()

        count = batch["image"].size(0)
        self.loss_meter_train.update(loss.item(), count)

        # Log loss and loss_meter per training step
        self.log_dict({'train_loss': self.train_loss, 'train_loss_meter': self.loss_meter_train.avg}, on_step=True)

        return loss

    def on_train_epoch_end(self):

        # Log loss and loss_meter
        self.log_dict({'train_loss': self.train_loss, 'train_loss_meter': self.loss_meter_train.avg}, on_epoch=True)

    def validation_step(self, batch):
        batch = {k: v.cuda() for k, v in batch.items()}
        # Get loss from the model
        loss = self.forward(batch)
        self.valid_loss = loss.item()
        
        count = batch["image"].size(0)
        self.loss_meter_valid.update(loss.item(), count)
        return loss

    def on_validation_epoch_end(self):
        print('Step: ', self.global_step)
        print('valid_loss: ', self.valid_loss)
        print('best_loss_valid: ', self.best_loss_valid)
        if self.valid_loss < self.best_loss_valid:
            self.best_loss_valid = self.valid_loss
            self.best_step = self.global_step

            print(f"Best model loss in step {self.global_step}")
        
        # Log metrics and best metrics in each validation step
        self.log_dict({'val_loss': self.valid_loss, 
                       'val_loss_meter': self.loss_meter_valid.avg,
                       'best_val_loss': self.best_loss_valid})

    def test_step(self, batch):
        batch = {k: v.cuda() for k, v in batch.items()}

        # Save gene expression masks
        self.test_masks.append(batch['mask'])

        # Get test images embeddings
        self.test_image_embeddings.append(self.get_image_embeddings(batch))

        # Get test gene expression embeddings and true value
        self.get_spot_embeddings(batch)

    def on_test_epoch_end(self):
        # Concatenate prediction masks
        self.test_masks = torch.cat(self.test_masks)

        img_embeddings_test = torch.cat(self.test_image_embeddings)
        spot_embeddings_test = torch.cat(self.test_spot_embeddings)
        spot_expressions_test = torch.cat(self.test_spot_expressions)

        self.get_spot_embeddings_train()

        # Find nearest neighbors in latent space
        print("Finding matches, using average of top 50 expressions")
        indices = self.find_matches(self.train_spot_embeddings, img_embeddings_test, top_k=50)
        indices = torch.tensor(indices).cuda()

        # Retrieve the gene embeddings and expressions of the nearest neighbors
        matched_spot_embeddings_pred = torch.zeros((indices.shape[0], self.train_spot_embeddings.shape[1])).cuda()
        matched_spot_expression_pred = torch.zeros((indices.shape[0], self.train_spot_expressions.shape[1])).cuda()
        for i in range(indices.shape[0]):
            matched_spot_embeddings_pred[i,:] = torch.mean(self.train_spot_embeddings[indices[i,:],:], axis=0)
            matched_spot_expression_pred[i,:] = torch.mean(self.train_spot_expressions[indices[i,:],:], axis=0)

        # Get metrics
        gt = spot_expressions_test
        pred = matched_spot_expression_pred
        metrics = get_metrics(gt, pred, mask=self.test_masks)
        
        # Put test prefix in metric dict
        metrics = {f'test_{k}': v for k, v in metrics.items()}
        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        try:
            optimizer = getattr(torch.optim, self.args.optimizer)(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        except:
            optimizer = getattr(torch.optim, self.args.optimizer)(self.parameters(), lr=self.args.lr)

        return optimizer

class CLIPModel_ViT(nn.Module):
    def __init__(
        self,
        spot_embedding,
        temperature=CFG.temperature,
        image_embedding=768
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT()
        #self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        #spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CLIPModel_CLIP(nn.Module):
    def __init__(
        self,
        spot_embedding,
        temperature=CFG.temperature,
        image_embedding=768
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_CLIP()
        #self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        #spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_ViT_L(nn.Module):
    def __init__(
        self,
        spot_embedding,
        temperature=CFG.temperature,
        image_embedding=1024
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_ViT_L()
        #self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) #aka the input dim, 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        #spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class CLIPModel_resnet101(nn.Module):
    def __init__(
        self,
        spot_embedding,
        temperature=CFG.temperature,
        image_embedding=2048
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet101()
        #self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) 
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        #spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

class CLIPModel_resnet152(nn.Module):
    def __init__(
        self,
        spot_embedding,
        temperature=CFG.temperature,
        image_embedding=2048
    ):
        super().__init__()
        self.image_encoder = ImageEncoder_resnet152()
        #self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding) 
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding) #3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]
        #spot_features = self.spot_encoder(batch["reduced_expression"])
        
        # Getting Image and Spot Embeddings (with same dimension) 
        image_embeddings = self.image_projection(image_features)
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(
            (images_similarity + spots_similarity) / 2 * self.temperature, dim=-1
        )
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")