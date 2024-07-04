import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from typing import List, Union

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False  # type: ignore

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def rle_decode(mask_rle: Union[str, int], shape=(224, 224)) -> np.array:
    if mask_rle == -1:
        return np.zeros(shape)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7) -> float:
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

def calculate_dice_scores(gt_mask_rle, pred_mask_rle, img_shape=(224, 224)) -> List[float]:
    def calculate_dice(pred_rle, gt_rle):
        pred_mask = rle_decode(pred_rle, img_shape)
        gt_mask = rle_decode(gt_rle, img_shape)

        if np.sum(gt_mask) > 0 or np.sum(pred_mask) > 0:
            return dice_score(pred_mask, gt_mask)
        else:
            return None  # No valid masks found, return None

    dice_scores = Parallel(n_jobs=-1)(
        delayed(calculate_dice)(pred_rle, gt_rle) for pred_rle, gt_rle in zip(pred_mask_rle, gt_mask_rle)
    )

    dice_scores = [score for score in dice_scores if score is not None]  # Exclude None values

    return np.mean(dice_scores)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1e-7):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        optimizer.zero_grad()
        images = images.float().to(device)
        masks = masks.float().to(device)
        outputs = model(images)
        loss = loss_fn(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()   

    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    mask_rles = []
    pred_mask_rles = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.float().to(device)
            masks = masks.float().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks.unsqueeze(1))
            total_loss += loss.item()

            pred_mask = torch.sigmoid(outputs).cpu().numpy()
            pred_mask = np.squeeze(pred_mask, axis=1)
            pred_mask = (pred_mask > 0.35).astype(np.uint8)
            
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i].cpu().numpy())
                mask_rles.append(mask_rle)
                
                pred_mask_rle = rle_encode(pred_mask[i])    
                if pred_mask_rle == '':
                    pred_mask_rles.append(-1)
                else:
                    pred_mask_rles.append(pred_mask_rle)
                    
    dice_score = calculate_dice_scores(mask_rles, pred_mask_rles)

    return total_loss / len(dataloader), dice_score

def infer(model, dataloader, device):
    model.eval()
    result = []

    with torch.no_grad():
        for images in dataloader:
            images = images.float().to(device)
            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8)
        
            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':
                    result.append(-1)
                else:
                    result.append(mask_rle)

    return result
