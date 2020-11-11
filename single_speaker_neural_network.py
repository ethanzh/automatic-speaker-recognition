import os
import random
import shutil
import sys
from math import ceil
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from pydub import AudioSegment
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.utils import class_weight
from tensorflow.python.keras import backend as K
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel

import wandb

class ClassifierDataset(Dataset):
    """Load numpy files from directory structure where each numpy file represents
    the extracted features from the pre-trained model"""

    def __init__(self, target_speaker, dataset_dir, non_target_clip_count_multiplier):
        """
        Args:
            target_speaker (str): path to target speaker split directory
            dataset_dir (str)
        """

        outputs = []
        labels = []

        for clip in os.listdir(f"{dataset_dir}/{target_speaker}"):
            if 'npy' not in clip:
                continue

            output = np.load(f'{dataset_dir}/{target_speaker}/{clip}')

            outputs.append(output)
            labels.append(1)

        num_target_clips = len(outputs)
        #print(f'INFO: Loaded {num_target_clips} clips for speaker {target_speaker}')
        all_speakers = [s for s in os.listdir(dataset_dir) if s != '.DS_Store']
        num_other_clips = min(num_target_clips * non_target_clip_count_multiplier, len(all_speakers))
        sampled_speakers = random.sample(all_speakers, num_other_clips)
        for speaker in sampled_speakers: 
            for clip in os.listdir(f"{dataset_dir}/{speaker}"):
                if 'npy' not in clip:
                    continue

                output = np.load(f"{dataset_dir}/{speaker}/{clip}")

                outputs.append(output)
                labels.append(0)

        self.outputs = np.array(outputs)
        self.labels = np.array(labels)

    def get_clip_counts(self):
        len_target_clips = len([x for x in self.labels if x == 1])
        len_other_clips = len(self.outputs) - len_target_clips

        return len_target_clips, len_other_clips

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.outputs[idx], self.labels[idx]


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.fc1(x)

        return torch.sigmoid(x)


def test_classifier(classifier, loader):
    all_labels = []
    all_predicted = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier.eval()
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            dimension = int(labels.shape[0])
            labels = labels.view(dimension, -1).float()

            outputs = classifier(inputs)
            outputs = outputs.to(device)

            all_labels += [int(x) for x in labels]

            all_predicted += [list(elems).index(max(elems)) for elems in outputs]

    return round(balanced_accuracy_score(all_labels, all_predicted, adjusted=True), 3)


def main(target_speaker=None, source_dir=None, use_checkpoint=False, num_epochs=200, batch_size=16, learning_rate=0.003, train_split=0.8, random_seed=1234, non_target_clip_count_multiplier=50):
    if not target_speaker:
        raise Exception('Must specify a target speaker')
    if not source_dir:
        raise Exception('Must specify a source dir')

    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print('INFO: Using device:', device)

    AUDIO_PATH = 'audio'
    SOURCE_DIR = source_dir  #'accents_features'
    MODEL_PATH = 'model.pt'
    USE_CHECKPOINT = use_checkpoint
    BATCH_SIZE = batch_size 
    NUM_EPOCHS = num_epochs
    TRAIN_SPLIT = train_split 
    LEARNING_RATE = learning_rate 

    dataset_dir = f"{AUDIO_PATH}/{SOURCE_DIR}"
    dataset = ClassifierDataset(target_speaker, dataset_dir, non_target_clip_count_multiplier=non_target_clip_count_multiplier)
    target_speaker_clips, other_speaker_clips = dataset.get_clip_counts()
    #print(f'INFO: Target speaker clip count: {target_speaker_clips}')
    #print(f'INFO: Other speaker clip count: {other_speaker_clips}')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    y_true = ([0] * other_speaker_clips) + ([1] * target_speaker_clips)

    weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=[0, 1],
                                            y=y_true)
    weights = torch.FloatTensor(weights)

    #print(f'INFO: Class weights: {weights}')

    classifier = Classifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    trained_classifier = train(classifier, optimizer, criterion, train_loader, val_loader, NUM_EPOCHS, device, BATCH_SIZE, MODEL_PATH, weights, wandb=False)


def train(classifier, optimizer, criterion, training_loader, validation_loader, num_epochs, device, batch_size, model_path, class_weights, wandb=False):
    if wandb:
        wandb.init(project='automatic-speaker-recognition')
        wandb.watch(classifier)

    initial_epoch_count = 0

    negative_weight = class_weights[0]
    positive_weight = class_weights[1]

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0

        for batch_index, (inputs, labels) in enumerate(training_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = classifier(inputs)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_index % 120 == 119:
                msg = f"INFO: [{initial_epoch_count + epoch + 1}, {batch_index + 1}]: train loss: {running_loss / 120}"
                print(msg)
                if wandb:
                    wandb.log({'train_loss': running_loss / 120})
                running_loss = 0.0

        classifier.eval()
        validation_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(validation_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = classifier(inputs)
            outputs = outputs.to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            validation_loss += loss.item()
            if batch_index % 120 == 119:
                msg = f"INFO: [{initial_epoch_count + epoch + 1}, {batch_index + 1}]: val loss: {validation_loss / 120}"
                print(msg)
                if wandb:
                    wandb.log({'validation_loss': validation_loss / 120})
                validation_loss = 0.0


        accuracy = test_classifier(classifier, validation_loader) 
        if epoch == 49:
            print(f'INFO: Accuracy at epoch {epoch}: {accuracy}')

        torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
            model_path,
        )

        if wandb:
            wandb.save(model_path)

if __name__ == '__main__':
    counts = [1, 5, 10, 20, 50, 100]
    for count in counts:
        print(count)
        main('ethan', 'sherlock_holmes_features', num_epochs=50, non_target_clip_count_multiplier=count)