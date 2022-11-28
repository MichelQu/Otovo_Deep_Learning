#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.path import Path

import pickle
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import cat
from torchvision import transforms
from sklearn.utils import gen_batches


# # Data Importation and Treatment

# In[2]:


# Importation of the data
filename = './data/meta_data.hdf'
row_data = pd.read_hdf(filename,'/d')
row_data2 = row_data.sort_values(by = 'building_id')[:3000]

print(f'There are {row_data.shape[0]} observations and each observation has {row_data.shape[1]} attributes.')


# In[3]:


# Here we merge all data (csv) together from the validation part 
all_files = ['paths_df_validated_0_to_6500_8000_to_8499.csv',
             'paths_df_validated_3000_5500.csv', 
             'paths_df_validated_5500_6500.csv']
data = pd.DataFrame()
isMatch = True


for file in all_files : 
    if file == 'paths_df_validated_0_to_6500_8000_to_8499.csv' : 
        temp = pd.read_csv(file)[:3000]
        temp = temp.loc[temp['valid'] == True]
        index_name = temp['original_index'].values
        data = pd.concat([data,row_data2.iloc[index_name]]) 
        # Check if the dataset is good 
        a = []
        for name in temp['image_path'].values : 
            a.append(name.replace('-b15-otovowms.jpeg',''))
        a = np.array(a)
        n_match = (row_data2.iloc[index_name]['building_id'].values == a).sum()
        if (n_match == row_data2.iloc[index_name].shape[0]) :
            isMatch = isMatch*True
        else :
            isMatch = isMatch*False
                    
    else :
        temp = pd.read_csv(file)
        temp = temp.loc[temp['valid'] == True] #Select only the 'True' = Validated images
        index_name = temp['original_index'].values
        data = pd.concat([data,row_data.iloc[index_name]])
        n_match = (temp['mask_path'].values == row_data.iloc[index_name]['building_id'].values).sum()
        if (n_match == row_data.iloc[index_name].shape[0]) :
            isMatch = isMatch*True
        else :
            isMatch = isMatch*False


data.drop_duplicates(subset=['building_id']) # Drop duplicate of images 

print(f'There are {data.shape[0]} validated observations and each observation has {data.shape[1]} attributes.')
if isMatch :
    print('All is good with the dataset, we can continue !')
else : 
    print('There is a mismatch between pictures (pictures names)')


# In[6]:


# This function creates the mask for one image, the mask is a array (500*500)
def mask_creation (n,p,surfaces) : 
    # If one polygon in surface list
    if (len(surfaces[0])==2) : 
        x, y = np.meshgrid(np.arange(n), np.arange(p))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        path = Path(surfaces)
        grid = path.contains_points(points)
        grid = grid.reshape((n,p))
    # If multiple polygones in surface list
    else :
        grid = np.reshape([False for i in range (n*p)],(n,p))
        for i,polygons in enumerate(surfaces) : 
            x, y = np.meshgrid(np.arange(n), np.arange(p))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            path = Path(polygons)
            grid_2 = path.contains_points(points)
            grid_2 = grid_2.reshape((n,p))
            grid = np.add(grid,grid_2)
    # We return the 500*500 matrix filled with True and False
    return (grid)



# This function creates a format that will be use for the DL Network
def creation_useful_dataset(dataset) : 
    # Creation of variables 
    X=[]; Y=[];
    length,_ = dataset.shape
    n = 500; p = 500; # Image Shape
    for i in tqdm(range(length)) :
        # Import the wanted picture and the matrix
        info = dataset.iloc[i]
        image_name = info.building_id + '-b15-otovowms.jpeg'
        im_matrix = plt.imread('./data/' + image_name)
        # We build the X_dataset 
        X.append(im_matrix)
        # We build the target_dataset
        surfaces = info.b_surfaces
        Y.append(mask_creation(n,p,surfaces))
    # We convert the array in pytorch Tensor 
    X = torch.Tensor(np.array(X))
    X = torch.swapaxes(X, 1, -1)  # To have (B,C,H,W)
    Y = torch.Tensor(np.array(Y)) 
    return (X,Y)


# **Creation of the training, validation and testing datasets**

# In[ ]:


# We create a split 70% Training set, 15% validation set, 15% testing set 
n_split1 = int(data.shape[0]*0.70) 
n_split2 = int(data.shape[0]*0.85)
df_train = data.iloc[:n_split1]
df_validation = data.iloc[n_split1:n_split2]
df_test = data.iloc[n_split2:]

print('Importation of the Training set')
X_train, Y_train = creation_useful_dataset(df_train)
print('Importation of the Validation set')
X_validation, Y_validation = creation_useful_dataset(df_validation)
print('Importation of the Testing set')
X_test, Y_test = creation_useful_dataset(df_test)


# In[ ]:


pickled = True

if pickled :
    # Dump the data into a pickle file
    with open('./pickles/data.X_train', 'wb') as f:
         pickle.dump(X_train, f)
    with open('./pickles/data.Y_train', 'wb') as f:
         pickle.dump(Y_train, f)
    with open('./pickles/data.X_validation', 'wb') as f:
         pickle.dump(X_validation, f)
    with open('./pickles/data.Y_validation', 'wb') as f:
         pickle.dump(Y_validation, f)
    with open('./pickles/data.X_test', 'wb') as f:
         pickle.dump(X_test, f)
    with open('./pickles/data.Y_test', 'wb') as f:
         pickle.dump(Y_test, f)


# # Data Augmentation
# 
# We can shift the image 90°,180°,270° to have more images to train on.

# In[111]:


def data_augmentation_function(data) :     
    
    def shift_function(x,y, angle) : 
        # Function what will create a new set of coordinates in function of the rotation
        if angle == 0 :
            return[x,y]
        if angle == 90 :
            return [y,500-x]
        if angle == 180 :
            return [500-x,500-y]
        else :
            return [500-y,x]
    
    # Some useful variables
    length_data = data.shape[0]
    Names = []; Angles = []; Surfaces = [];
    different_angles = [0,90,180,270]
    
    # Creation of the shifted images
    print('------- Creation of Shifted Data ---------')
    for i in tqdm(range(length_data)) :
        image = data.iloc[i]
        image_name = image.building_id +  '-b15-otovowms.jpeg'
        image_matrix = plt.imread('./data/' + image_name)
        surface = image.b_surfaces
        Names.append([image_name,image_name,image_name,image_name])
        Angles.append(different_angles)
        temp_surf = []
        for j, angle in enumerate (different_angles) :
            new_surfaces = []
            # If there is only one polygon in surface
            if (len(surface[0])==2) :
                for point in surface :
                    x,y = point
                    new_surfaces.append(shift_function(x,y,angle))
            # Multiples polygones
            else : 
                for polygon in surface :
                    new_polygon = [] 
                    for point in polygon :
                        x,y = point
                        new_polygon.append(shift_function(x,y,angle))
                    new_surfaces.append(new_polygon)
            temp_surf.append(new_surfaces)
        Surfaces.append(temp_surf)
    
    # Reshape  
    Names = np.array(Names).reshape(-1)
    Angles = np.array(Angles).reshape(-1)
    Surfaces = np.array(Surfaces).reshape(-1)
    # Creation of the shifted dataset 
    data_augmentation = pd.DataFrame()
    data_augmentation['building_id'] = Names
    data_augmentation['shift_angle'] = Angles
    data_augmentation['b_surfaces'] = Surfaces
    
    def creation_useful_datasetV2(dataset,length_data) : 
        X_data = [];Y_data = [];
        for j in tqdm (range(length_data)) : 
            for i in range (4) :
                image = data_augmentation.iloc[i+j*4]
                im_name = image.building_id
                if i == 0 : 
                    im_matrix = plt.imread('./data/' + im_name)
                else : 
                    im_matrix = [list(reversed(t)) for t in zip(*im_matrix)]
                # Creation of X,y dataset 
                X_data.append(im_matrix)
                surfaces = image.b_surfaces
                Y_data.append(mask_creation(500,500,surfaces))
        # We convert the array in pytorch Tensor 
        X_data = torch.Tensor(np.array(X_data))
        X_data = torch.swapaxes(X_data, 1, -1)  # To have (B,C,H,W)
        Y_data = torch.Tensor(np.array(Y_data))
        return (X_data,Y_data)

    # Creation of the X and the Y dataset
    print('------- Creation of the useful Dataset ---------')
    X_data, Y_data = creation_useful_datasetV2(data_augmentation,length_data)
    return(X_data,Y_data)


# In[112]:


# We create a split 70% Training set, 15% validation set, 15% testing set 
n_split1 = int(data.shape[0]*0.70) 
n_split2 = int(data.shape[0]*0.85)
df_train = data.iloc[:n_split1]
df_validation = data.iloc[n_split1:n_split2]
df_test = data.iloc[n_split2:]

print('Importation of the augmented Training set')
X_train, Y_train = data_augmentation_function(df_train)
print('Importation of the augmented Validation set')
X_validation, Y_validation = data_augmentation_function(df_validation)
print('Importation of the augmented Testing set')
X_test, Y_test = data_augmentation_function(df_test)


# In[ ]:


pickled = True

if pickled :
    # Dump the data into a pickle file
    with open('./pickles/data.X_train', 'wb') as f:
         pickle.dump(X_train, f)
    with open('./pickles/data.Y_train', 'wb') as f:
         pickle.dump(Y_train, f)
    with open('./pickles/data.X_validation', 'wb') as f:
         pickle.dump(X_validation, f)
    with open('./pickles/data.Y_validation', 'wb') as f:
         pickle.dump(Y_validation, f)
    with open('./pickles/data.X_test', 'wb') as f:
         pickle.dump(X_test, f)
    with open('./pickles/data.Y_test', 'wb') as f:
         pickle.dump(Y_test, f)


# # Deep Learning Network 

# In[19]:


pickled = False 

if pickled :
    # Load the data from a pickle file
    with open('./pickles/data.X_train', 'rb') as f:
         X_train = pickle.load(f)
    with open('./pickles/data.Y_train', 'rb') as f:
         Y_train = pickle.load(f)
    with open('./pickles/data.X_validation', 'rb') as f:
         X_validation = pickle.load(f)
    with open('./pickles/data.Y_validation', 'rb') as f:
         Y_validation = pickle.load(f)
    with open('./pickles/data.X_test', 'rb') as f:
         X_test = pickle.load(f)
    with open('./pickles/data.Y_test', 'rb') as f:
         Y_test = pickle.load(f)


# In[ ]:


print(f'X_train : {X_train.shape} - [Batchsize,Channel,Height,Width]')
print(f'Y_train : {Y_train.shape} - [Batchsize,Height,Width]')
print(f'X_validation : {X_validation.shape} - [Batchsize,Channel,Height,Width]')
print(f'Y_validation : {Y_validation.shape} - [Batchsize,Height,Width]')
print(f'X_test : {X_test.shape} - [Batchsize,Channel,Height,Width]')
print(f'Y_test : {Y_test.shape} - [Batchsize,Height,Width]')


# In[108]:


# The convolutional network (U-net)
class UNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Contracting Path
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv31 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv41 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv51 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Expanding Path
        self.up5 = nn.Upsample(size=(62,62))
        self.conv53 = nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        
        self.conv61 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.conv62 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.up6 = nn.Upsample(size=(125,125))
        self.conv63 = nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv71 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.conv72 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.up7 = nn.Upsample(size=(250,250))
        self.conv73 = nn.Conv2d(in_channels = 256, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        self.conv81 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.conv82 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.up8 = nn.Upsample(size=(500,500))
        self.conv83 = nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.conv91 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.conv92 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.conv93 = nn.Conv2d(in_channels=64,out_channels=2,kernel_size=3, stride=1, padding=1)

        # Output
        self.conv10 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1)

    def forward(self, x):
        # Contracting Path
        x1 = F.relu(self.conv11(x))
        x1 = F.relu(self.conv12(x1))
        
        x2 = self.pool1(x1)
        x2 = F.relu(self.conv21(x2))
        x2 = F.relu(self.conv22(x2))
        
        x3 = self.pool2(x2)
        x3 = F.relu(self.conv31(x3))
        x3 = F.relu(self.conv32(x3))
        
        x4 = self.pool3(x3)
        x4 = F.relu(self.conv41(x4))
        x4 = F.relu(self.conv42(x4))
        
        x5 = self.pool4(x4)
        x5 = F.relu(self.conv51(x5))
        x5 = F.relu(self.conv52(x5))

        # Expanding Path
        x5 = self.up5(x5)
        x6 = self.conv53(x5)
        x4 = transforms.CenterCrop(62)(x4)
        x6 = cat([x4, x6], axis=1)
        x6 = F.relu(self.conv61(x6))
        x6 = F.relu(self.conv62(x6))
        
        x6 = self.up6(x6)
        x7 = self.conv63(x6)
        x3 = transforms.CenterCrop(125)(x3)
        x7 = cat([x3, x7], axis=1)
        x7 = F.relu(self.conv71(x7))
        x7 = F.relu(self.conv72(x7))
        
        x7 = self.up7(x7)
        x8 = self.conv73(x7)
        x2 = transforms.CenterCrop(250)(x2)
        x8 = cat([x2, x8], axis=1)
        x8 = F.relu(self.conv81(x8))
        x8 = F.relu(self.conv82(x8))
        
        x8 = self.up8(x8)
        x9 = self.conv83(x8)
        x1 = transforms.CenterCrop(500)(x1)
        x9 = cat([x1, x9], axis=1)      
        x9 = F.relu(self.conv91(x9))
        x9 = F.relu(self.conv92(x9))
        x9 = F.relu(self.conv93(x9))

        # Output
        x10 = self.conv10(x9)
        x10 = torch.sigmoid(x10)
        return x10


# In[109]:


class UNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        # Contracting Path
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3, stride=1, padding=1)        
        self.bn11 = nn.BatchNorm2d(num_features = 64, eps = 0.00001, momentum = 0.1)
        self.conv12 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features = 64, eps = 0.00001, momentum = 0.1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv21 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(num_features = 128, eps = 0.00001, momentum = 0.1)
        self.conv22 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(num_features = 128, eps = 0.00001, momentum = 0.1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv31 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(num_features = 256, eps = 0.00001, momentum = 0.1)
        self.conv32 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.bn32 = nn.BatchNorm2d(num_features = 256, eps = 0.00001, momentum = 0.1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv41 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(num_features = 512, eps = 0.00001, momentum = 0.1)
        self.conv42 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.bn42 = nn.BatchNorm2d(num_features = 512, eps = 0.00001, momentum = 0.1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv51 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3, stride=1, padding=1)
        self.bn51 = nn.BatchNorm2d(num_features = 1024, eps = 0.00001, momentum = 0.1)
        self.conv52 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3, stride=1, padding=1)
        self.bn52 = nn.BatchNorm2d(num_features = 1024, eps = 0.00001, momentum = 0.1)
        self.pool5 = nn.MaxPool2d(2, 2)

        # Expanding Path
        self.up5 = nn.Upsample(size=(62,62))
        self.conv53 = nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn53 = nn.BatchNorm2d(num_features = 512, eps = 0.00001, momentum = 0.1)
        
        self.conv61 = nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.bn61 = nn.BatchNorm2d(num_features = 512, eps = 0.00001, momentum = 0.1)
        self.conv62 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, stride=1, padding=1)
        self.bn62 = nn.BatchNorm2d(num_features = 512, eps = 0.00001, momentum = 0.1)
        self.up6 = nn.Upsample(size=(125,125))
        self.conv63 = nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn63 = nn.BatchNorm2d(num_features = 256, eps = 0.00001, momentum = 0.1)

        self.conv71 = nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.bn71 = nn.BatchNorm2d(num_features = 256, eps = 0.00001, momentum = 0.1)
        self.conv72 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, stride=1, padding=1)
        self.bn72 = nn.BatchNorm2d(num_features = 256, eps = 0.00001, momentum = 0.1)
        self.up7 = nn.Upsample(size=(250,250))
        self.conv73 = nn.Conv2d(in_channels = 256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn73 = nn.BatchNorm2d(num_features = 128, eps = 0.00001, momentum = 0.1)
        
        self.conv81 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn81 = nn.BatchNorm2d(num_features = 128, eps = 0.00001, momentum = 0.1)
        self.conv82 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, stride=1, padding=1)
        self.bn82 = nn.BatchNorm2d(num_features = 128, eps = 0.00001, momentum = 0.1)
        self.up8 = nn.Upsample(size=(500,500))
        self.conv83 = nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn83 = nn.BatchNorm2d(num_features = 64, eps = 0.00001, momentum = 0.1)
        
        self.conv91 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.bn91 = nn.BatchNorm2d(num_features = 64, eps = 0.00001, momentum = 0.1)
        self.conv92 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3, stride=1, padding=1)
        self.bn92 = nn.BatchNorm2d(num_features = 64, eps = 0.00001, momentum = 0.1)
        self.conv93 = nn.Conv2d(in_channels=64,out_channels=2,kernel_size=3, stride=1, padding=1)
        self.bn93 = nn.BatchNorm2d(num_features = 2, eps = 0.00001, momentum = 0.1)

        # Output
        self.conv10 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1)
        self.bn10 = nn.BatchNorm2d(num_features = 1, eps = 0.00001, momentum = 0.1)
        
        # The Dropout Rate of the Network
        self.dropout = nn.Dropout(0.2) 

    def forward(self, x):
        # Contracting Path
        x1 = F.relu(self.bn11(self.conv11(x)))
        x1 = F.relu(self.bn12(self.conv12(x1)))
        x1 = self.pool1(x1)
        x2 = self.dropout(x1)
        
        x2 = F.relu(self.bn21(self.conv21(x2)))
        x2 = F.relu(self.bn22(self.conv22(x2)))
        x2 = self.pool2(x2)
        x3 = self.dropout(x2)
        
        x3 = F.relu(self.bn31(self.conv31(x3)))
        x3 = F.relu(self.bn32(self.conv32(x3)))
        x3 = self.pool3(x3)
        x4 = self.dropout(x3)
        
        x4 = F.relu(self.bn41(self.conv41(x4)))
        x4 = F.relu(self.bn42(self.conv42(x4)))
        x4 = self.pool4(x4)
        x5 = self.dropout(x4)
        
        x5 = F.relu(self.bn51(self.conv51(x5)))
        x5 = F.relu(self.bn52(self.conv52(x5)))

        # Expanding Path
        x5 = self.up5(x5)
        x6 = self.bn53(self.conv53(x5))
        x4 = transforms.CenterCrop(62)(x4)
        x6 = cat([x4, x6], axis=1)
        x6 = self.dropout(x6)
        x6 = F.relu(self.bn61(self.conv61(x6)))
        x6 = F.relu(self.bn62(self.conv62(x6)))
        
        x6 = self.up6(x6)
        x7 = self.bn63(self.conv63(x6))
        x3 = transforms.CenterCrop(125)(x3)
        x7 = cat([x3, x7], axis=1)
        x7 = self.dropout(x7)
        x7 = F.relu(self.bn71(self.conv71(x7)))
        x7 = F.relu(self.bn72(self.conv72(x7)))
        
        x7 = self.up7(x7)
        x8 = self.bn73(self.conv73(x7))
        x2 = transforms.CenterCrop(250)(x2)
        x8 = cat([x2, x8], axis=1)
        x8 = self.dropout(x8)
        x8 = F.relu(self.bn81(self.conv81(x8)))
        x8 = F.relu(self.bn82(self.conv82(x8)))
        
        x8 = self.up8(x8)
        x9 = self.bn83(self.conv83(x8))
        x1 = transforms.CenterCrop(500)(x1)
        x9 = cat([x1, x9], axis=1)      
        x9 = self.dropout(x9)
        x9 = F.relu(self.bn91(self.conv91(x9)))
        x9 = F.relu(self.bn92(self.conv92(x9)))
        x9 = F.relu(self.bn93(self.conv93(x9)))

        # Output
        x10 = self.bn10(self.conv10(x9))
        x10 = torch.sigmoid(x10)
        return x10


# Use : 
# 
#       Binary Cross Entropy, BCE-Dice Loss, Binary Cross Entropy
#       
#       Optimizer : Adam, RMSProp
#       
#       Epoch : Around 20

# In[51]:


# Settings of the model 
net1 = UNetV2()      # Simple model
net2 = UNetV3()     # Model with dropout and BatchNorm2d
models = [net1,net2]

# Batch Creation for the training of the model
batch_size = 64
batches_index_train = list(gen_batches(X_train.shape[0],batch_size=batch_size))
n_epoch = 20

for j,net in enumerate(models) : 
    # Parameters of the CNN model 
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()
    # Someful paramaters 
    index_of_validation = []
    losses_values_per_batches = []
    
    print(f'Model n°{j}')
    for epoch in range(n_epoch) : # loop over the dataset multiple times
        print(f'   Epoch n°{epoch+1}/{n_epoch}')
        for i in tqdm(range(len(batches_index_train))) : 
            # Take the batch we use
            batch_slice = batches_index_train[i]                         # Get the index of the batch
            inputs, masks = X_train[batch_slice], Y_train[batch_slice]   # Get inputs and masks
            # Training of the model
            optimizer.zero_grad()
            outputs = torch.squeeze(net(inputs),dim=1)     # To reduce the dimension to [B,H,W] of outputs
            loss = criterion(outputs, masks)               # Compute the loss value 
            loss.backward()                                # Backward of loss function
            optimizer.step()                               # We optimize the network
            
            if ((epoch*len(batches_index_train)+i)%5 == 0) : #Every 5 batches
                # We compute loss for this batch on the validation set
                net.eval()                                     # We want to evaluate the model
                inputs, masks = X_validation, Y_validation     # Get the inputs and masks of validation set 
                outputs = torch.squeeze(net(inputs),dim=1)     # Output of the model    
                loss = criterion(outputs, masks)               # Compute the loss value 
                index_of_validation.append(epoch*len(batches_index_train)+i) # The batch number
                losses_values_per_batches.append(loss.mean())  # Append the mean loss of the batch 
                net.train()                                    # We turn the model to training mode 
    
    # Plotting stuffs
    name = f'./pickles/cnn_model_{j}'
    with open(name, 'wb') as f:
        pickle.dump(net, f)
    plt.plot(index_of_validation,[float(x) for x in losses_values_per_batches])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per epoch')
    plt.savefig(f'./pickles/graph_loss_per_batch_{j}.png')
    plt.plot()


# # Importation of the DL network

# In[127]:


# Pickle the model 
with open('./pickles/cnn_model', 'rb') as f:
     net = pickle.load(f)
with open('./pickles/cnn_model_improved', 'rb') as f:
     net_improved = pickle.load(f)


# In[128]:


# Testing the model 
outputs = torch.squeeze(net(X_test),dim=1)
outputs = (outputs>0.5)
print(f'The loss is : {criterion(outputs, Y_test)}')


# In[129]:


# Plotting for rigolade 
for i, prediction in enumerate(outputs[:3]) :
    fig,ax = plt.subplots(1,2)
    true_mask = np.array(Y_test[i])
    prediction = np.array(prediction)
    ax[0].imshow(true_mask)
    ax[1].imshow(prediction)
    plt.show()


# # To do list :
# 
# **Optimizer :** We can try with different optimizers to see some differences.
# 
# **Batch :** Export and Import batch data for each epoch to use less memory.
# 
# **Metrics :** We can use intersection over union (IoU) with the masks, AUC, position recall curve.
# 
# **Expansion of the model :** We can test the model on unseen data to see if it can be generalized.
# 
# **Create a validation set :** In order to validate of model for each epoch, we can compute the loss withthe validation set.

# # Old Code

# In[104]:


# The loss function : Intersection over Union for validation set (IoU)
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # Intersection is equivalent to True Positive count
        # Union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        return 1 - IoU


# In[105]:


# This bloc plots the picture and the corresponding polygon surfaces.
showing = False 

# This function plots the surface polygons of the selected image 
def image_reader (image_data) : 
    surfaces = image_data.b_surfaces
    # One polygone
    if (len(surfaces[0])==2) : 
        plt.plot(np.array(surfaces)[:,1],np.array(surfaces)[:,0], label = 'surface', c = 'r')
    # Multiples polygones
    else :
        for polygons in surfaces : 
            plt.plot(np.array(polygons)[:,1],np.array(polygons)[:,0], label = 'surface', c = 'r')
    return 0

# The for-loop that will plot the pictures 
if showing :
    n = 1
    for i in range (n) : 
        a = data.iloc[i]
        im_name = a.building_id +  '-b15-otovowms.jpeg'
        print(im_name)
        im_matrix = plt.imread('./data/' + im_name)
        plt.imshow(im_matrix)
        image_reader(a)
        plt.show()


# In[115]:


small_dataset_for_test = True 

## For testing the network, we take a smaller dataset 
if small_dataset_for_test : 
    n_split = 3
    df_train = data.iloc[:n_split]
    df_validation = data.iloc[n_split:n_split+2]
    df_test = data.iloc[n_split+2:n_split+4]
    print('Importation of the Training set')
    X_train, Y_train = creation_useful_dataset(df_train)
    print('Importation of the Validation set')
    X_validation, Y_validation = creation_useful_dataset(df_validation)
    print('Importation of the Testing set')
    X_test, Y_test = creation_useful_dataset(df_test)


# In[125]:


def intersection_over_union(mask,prediction) :
    component1 = np.array(mask, dtype=bool)
    component2 = np.array(prediction, dtype=bool)

    overlap = component1*component2 # Logical AND
    union = component1 + component2 # Logical OR

    IOU = overlap.sum()/float(union.sum())
    return(IOU)

intersection_over_union(Y_train[0],Y_train[2])


# In[ ]:




