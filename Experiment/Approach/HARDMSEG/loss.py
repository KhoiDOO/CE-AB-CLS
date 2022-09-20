import torch 
import torch.nn.functional as F
import numpy as np

def structureloss(pred, mask):
	"""structureloss _summary_

	Arguments:
		pred -- Output tensor of shape (1, 1, width, height) representing the output of model
		mask -- Output tensor of shape (1, 1, width, height) representing the expected output of model

	Returns:
		Loss between prediction and mask
	"""
	weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
	wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

	pred = torch.sigmoid(pred)
	inter = ((pred * mask)*weit).sum(dim=(2, 3))
	union = ((pred + mask)*weit).sum(dim=(2, 3))
	wiou = 1 - (inter + 1)/(union - inter+1)
	return (wbce + wiou).mean()

pred = torch.empty(1, 1, 352, 352).random_(2)
mask = torch.empty(1, 1, 352, 352).random_(2)
print(np.unique(pred.numpy()))
print(np.unique(mask.numpy()))
print(pred.numpy().shape)
print(mask.numpy().shape)
print("Loss: {}".format(structureloss(pred, mask)))