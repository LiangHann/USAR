import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import erosion, dilation, cube
from scipy import stats

from sklearn.metrics import auc


def dice_coieffience_np_i(result_mask, x_mask):
    eps = 1e-5
    intersection =  np.sum(result_mask * x_mask)
    union =  eps + np.sum(result_mask) + np.sum(x_mask)
    dice = 2 * intersection/union
    return dice

def dice_coieffience_np(result_mask, x_mask):
    eps = 1e-5
    if len(result_mask.shape) + 1 == len(x_mask.shape):
        result_mask = np.expand_dims(result_mask, axis=-1)
    intersection =  np.sum(result_mask * x_mask)
    union =  eps + np.sum(result_mask) + np.sum(x_mask)
    dice = 2 * intersection/union
    return dice

def IOU_i(result_mask, x_mask):
    eps = 1e-5
    intersection =  np.sum(result_mask * x_mask)
    union =  np.sum(np.maximum(result_mask, x_mask))
    IOU = intersection/union
    return IOU

def IOU(result_mask, x_mask):
    eps = 1e-5
    if len(result_mask.shape) + 1 == len(x_mask.shape):
        result_mask = np.expand_dims(result_mask, axis=-1)
    intersection =  np.sum(result_mask * x_mask, axis=(1,2,3))
    union =  np.sum(np.maximum(result_mask, x_mask),axis=(1,2,3))
    IOU = np.mean(intersection/union)
    return IOU

def f1score(result_mask, x_mask):
    acc=(result_mask*x_mask).sum()/result_mask.sum()
    rec=(result_mask*x_mask).sum()/x_mask.sum()
    return 2*acc*rec/(acc+rec)

def iou2dice(x):
    return x/(x+(1-x)/2)

def aul(result_mask, x_mask):
    iternn =102
    dd = np.zeros(iternn)
    ff = np.zeros(iternn)
    for i in range(iternn-2):
        result_mask_i = (result_mask-float(i)/iternn+0.5).round()
        dd[i+1] = (result_mask_i*x_mask).sum()/result_mask_i.sum()
        ff[i+1] = (result_mask_i*x_mask).sum()/x_mask.sum()
    dd[0] = 0
    dd[-1] = 1
    ff[0] = 1
    ff[-1] = 0
    ff[ff>1]=1
    dd[dd>1]=1
    return auc(ff,dd)


def metrics(prediction, groundtruth):
    Dice_orig = []
    Dice_post = []
    Dice = []
    IOU = []
    F1score = []
    AUL = []
    N = prediction.shape[0]
    for i in range(N):
        dice_orig = dice_coieffience_np(prediction[i, :, :], groundtruth[i, :, :])
        Dice_orig += [dice_orig]
        prediction[i, :, :] = binary_fill_holes(prediction[i, :, :])
        dice_post = dice_coieffience_np(prediction[i, :, :], groundtruth[i, :, :])
        Dice_post += [dice_post]
        PD = prediction[i, :, :]
        GT = groundtruth[i, :, :]
        label_PD = label(PD)
        label_GT = label(GT)
        cells = regionprops(label_GT)
        for x in cells:
            p = label_PD[label_GT == x.label]
            if p.max() == 0:
                continue
            id2 = stats.mode(p[p>0])[0][0]
            idx = np.logical_or(label_PD == id2, label_GT==x.label)
            Dice += [dice_coieffience_np_i(PD[idx], GT[idx])]
            IOU += [IOU_i(PD[idx], GT[idx])]
            F1score += [f1score(PD[idx], GT[idx])]
            AUL += [aul(PD[idx], GT[idx])]

    return np.mean(Dice_orig), np.mean(Dice_post), np.mean(Dice), np.mean(IOU), np.mean(F1score), np.mean(AUL)

