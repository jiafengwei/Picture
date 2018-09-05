# Process raw data and save them into pickle file.
import os
import numpy as np
from PIL import Image
from PIL import ImageOps
from scipy import misc
import scipy.io
from skimage import io
import cv2
import sys
import cPickle as pickle
import glob
import random
from tqdm import tqdm
from eliaLib import dataRepresentation
from constants import *

img_size = INPUT_SIZE
salmap_size = INPUT_SIZE

# Resize train/validation files

listImgFiles = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToMaps, '*'))]

trainData = []
for currFile in tqdm(listImgFiles):
    trainData.append(dataRepresentation.Target(os.path.join(pathToImages, currFile + '.jpg'),
                                   os.path.join(pathToMaps, currFile + '.jpg'),                                 
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.image,
                                   dataRepresentation.LoadState.loaded, dataRepresentation.InputType.imageGrayscale))

    # if tt.image.getImage().shape[:2] != (480, 640):
    #    print 'Error:', currFile



# Resize test files

# LOAD DATA

# Train


with open(os.path.join(pathToPickle, 'minglangtrainData.pickle'), 'wb') as f:
    pickle.dump(trainData, f)

# Validation

