#!/usr/bin/env python
# coding: utf-8

# # Speaker Recognition - Wireless Health

# In[181]:


import os
import sys
import numpy as np
import random
from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import SAMPLE_RATE, NUM_FRAMES
from conv_models import DeepSpeakerModel
from test import batch_cosine_similarity
import tensorflow as tf
import logging


# In[182]:


is_cli = 'nn.py' in sys.argv[0]


# In[183]:


np.random.seed(1234)
random.seed(1234)


# In[184]:


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


# ## Import Pre-trained Model  

# In[185]:


if is_cli:
    print('Loading pre-trained model...')
model = DeepSpeakerModel()
model.m.load_weights('ResCNN_triplet_training_checkpoint_265.h5', by_name=True)


# ## Generate Features from Samples
# This simply discovers every training sample
# in the `samples` folder and retrieves a prediction
# from the pre-trained neural network to use as a
# feature vector for the below neural network.

# In[204]:


if is_cli:
    print('Loading data...')

directories = ['EthanHouston', 'LegalEagle', 'PhilippeRemy', 'MichelleObama']

train_test_split = 0.3

def generate_data(directories, train=True):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for i, directory in enumerate(directories):
        path = f'samples/{directory}'
        count = 0
        files = os.listdir(path)
        random.shuffle(files)
        for file_idx, file in enumerate(files):
            if '.wav' not in file:
                continue

            mfcc = sample_from_mfcc(read_mfcc(f'{path}/{file}', SAMPLE_RATE), NUM_FRAMES)
            predict = model.m.predict(np.expand_dims(mfcc, axis=0))
            
            label = [0] * len(directories)
            label[i] = 1
            
            if file_idx < int(len(files) * train_test_split):
                train_data.append(predict)
                train_labels.append(label)
            else:
                test_data.append(predict)
                test_labels.append(label)
            
            count += 1

        print(f'\tLoaded {count} file(s) from {path}')

    print(f'{len(train_data)} training samples, {len(test_data)} test'           f' samples ({round(100 * len(train_data) / len(test_data), 2)}%)')
        
    return train_data, train_labels, test_data, test_labels      
    
X_train, y_train, X_test, y_test = generate_data(directories)


# ## Train PyTorch Neural Network

# In[205]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[206]:


class Net(nn.Module):
    
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=1)

        return x


# In[207]:


import torch.optim as optim

print('Training...')

num_classes = len(directories)
net = Net(num_classes=num_classes)

num_epochs = 5000
lr = 0.003

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

for epoch_num, epoch in enumerate(range(num_epochs)):
    net.train()
    running_loss = 0.0

    for i, data in enumerate(zip(X_train, y_train)):
        inputs, labels = data

        optimizer.zero_grad()

        inputs = torch.from_numpy(inputs)
        outputs = net(inputs)

        labels = np.array(labels)
        idx = np.argmax(labels)
        labels = torch.from_numpy(np.array([idx]))
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# # Test the Network
# Specify which directories to discover test data
# within.

# In[214]:


if is_cli:
    print('Evaluating model...')

net.eval()
with torch.no_grad():
    correct = 0
    
    for train, label in zip(X_test, y_test):
        torch_data = torch.from_numpy(train)
        output = net(torch_data)
        predicted = torch.argmax(output)
        actual = np.argmax(label)

        if predicted == actual:
            correct += 1
        else:
            prediction_for_true_label = int(100 * output[0][actual])
            print('incorrect prediction,', f'{prediction_for_true_label}%')
            
    print(f'Testing accuracy: {int(100 * correct / len(X_test))}%')


# In[ ]:




