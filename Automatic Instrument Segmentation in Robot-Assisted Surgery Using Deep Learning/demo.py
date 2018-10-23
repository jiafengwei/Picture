%matplotlib inline
from pylab import *
import cv2

rcParams['figure.figsize'] = 10, 10

from dataset import load_image
import torch
from utils import cuda
from generate_masks import get_model
from albumentations import Compose, Normalize
from albumentations.torch.functional import img_to_tensor

def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)

	
def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img

model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')

img_file_name = 'data/cropped_train/instrument_dataset_3/images/frame004.jpg'
gt_file_name = 'data/cropped_train/instrument_dataset_3/binary_masks/frame004.png'

image = load_image(img_file_name)
gt = cv2.imread(gt_file_name, 0) > 0

imshow(image)

with torch.no_grad():
    input_image = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=image)['image']).cuda(), dim=0)
	
mask = model(input_image)
mask_array = mask.data[0].cpu().numpy()[0]

imshow(mask_array > 0)
imshow(mask_overlay(image, (mask_array > 0).astype(np.uint8)))