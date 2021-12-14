#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import time
import copy
from PIL import Image
import glob
import cv2



class detector:
		
	def __init__(self,filepath):
		self.filepath = filepath
		self.model = torch.load(filepath)
		self.class_names = ['with_mask',
		'with_mask_incorrect',
		 'without_mask'
		]
		

	def process_image(self,image):
	    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
		returns an Numpy array
	    '''
	    
	    pil_image = image
	   
	    image_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ])
	    
	    img = image_transforms(pil_image)
	    return img
    
	def classify_face(self,image):
	    device = torch.device("cpu")
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    im = Image.fromarray(image)
	    image = self.process_image(im)
	    print('image_processed')
	    img = image.unsqueeze_(0)
	    img = image.float()

	    self.model.eval()
	    self.model.to(device)
	    output = self.model(image)
	    print(output,'##############output###########')
	    _, predicted = torch.max(output, 1)
	    print(predicted.data[0],"predicted")


	    classification1 = predicted.data[0]
	    index = int(classification1)
	    print(self.class_names[index])
	    return self.class_names[index]









