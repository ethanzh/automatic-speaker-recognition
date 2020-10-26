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
from sklearn.metrics import f1_score, confusion_matrix, multilabel_confusion_matrix
from tensorflow.python.keras import backend as K
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel

import wandb


class ClassifierDataset(Dataset):
    """Load numpy files from directory structure where each numpy file represents
    the extracted features from the pre-trained model"""

    def __init__(self, in_speakers, out_speakers, dataset_dir):
        outputs = []
        labels = []

        for i, speaker in enumerate(in_speakers):
            for clip in os.listdir(f"{dataset_dir}/{speaker}"):
                if "npy" not in clip:
                    continue

                output = np.load(f"{dataset_dir}/{speaker}/{clip}")

                outputs.append(output)
                labels.append(i)

        none_index = len(set(labels))

        for speaker in out_speakers: 
            for clip in os.listdir(f"{dataset_dir}/{speaker}"):
                if "npy" not in clip:
                    continue

                output = np.load(f"{dataset_dir}/{speaker}/{clip}")

                outputs.append(output)
                labels.append(none_index)

        self.outputs = np.array(outputs)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return self.outputs[idx], self.labels[idx]


class Classifier(nn.Module):
    """Define a simple linear neural network

    Args:
        num_classes: the number of classes we are classifying

    """

    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=1)

        return x


# TODO: Is this the right data to test with?
def test_classifier(classifier, loader, count):
    #class_correct = [0] * count
    #class_total = [0] * count

    # used to calculate global f1
    all_labels = []
    all_predicted = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classifier.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = classifier(images)
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs, 1)

            all_labels += [int(x) for x in labels]
            all_predicted += [int(x) for x in predicted]

    weighted_f1 = f1_score(all_labels, all_predicted, average="weighted")
    micro_f1 = f1_score(all_labels, all_predicted, average="micro")
    macro_f1 = f1_score(all_labels, all_predicted, average="macro")

    return weighted_f1, micro_f1, macro_f1, all_labels, all_predicted


def main(use_checkpoint=False, in_speaker_ratio=0.8, num_epochs=200, batch_size=16, learning_rate=0.003, train_split=0.8, random_seed=1234):
    np.random.seed(random_seed)
    random.seed(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('INFO: Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('INFO: Memory Usage:')
        print('INFO: Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('INFO: Reserved:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    AUDIO_PATH = 'audio'
    SOURCE_DIR = 'accents_features'
    MODEL_PATH = 'model.pt'
    USE_CHECKPOINT = use_checkpoint
    BATCH_SIZE = batch_size 
    NUM_EPOCHS = num_epochs
    TRAIN_SPLIT = train_split 
    LEARNING_RATE = learning_rate 
    IN_SPEAKER_RATIO = in_speaker_ratio

    dataset_dir = f"{AUDIO_PATH}/{SOURCE_DIR}"
    all_speakers = [s for s in os.listdir(dataset_dir) if s != '.DS_Store']

    random.shuffle(all_speakers)

    in_speakers_index = int(len(all_speakers) * IN_SPEAKER_RATIO)
    in_speakers = all_speakers[:in_speakers_index]
    out_speakers = all_speakers[in_speakers_index:]

    # for weighting keep track of number of clips for each speaker
    speaker_clip_counts = []
    for in_speaker in in_speakers:
        clips = [c for c in os.listdir(f'{dataset_dir}/{in_speaker}') if c != '.DS_Store']
        speaker_clip_counts.append(len(clips))

    # sum all of the out_speaker clip counts for null classifier weight
    total_out_speaker_clips = 0
    for out_speaker in out_speakers:
        clips = [c for c in os.listdir(f'{dataset_dir}/{out_speaker}') if c != '.DS_Store']
        total_out_speaker_clips += len(clips)

    speaker_clip_counts.append(total_out_speaker_clips)
    total_clip_count = sum(speaker_clip_counts) + total_out_speaker_clips
    print(f'INFO: In speaker count: {len(in_speakers)}. Total clip count: {total_clip_count}')

    assert len(speaker_clip_counts) == len(in_speakers) + 1 

    full_dataset = ClassifierDataset(in_speakers, out_speakers, dataset_dir)

    # account for the null class
    NUM_CLASSES = len(in_speakers) + 1

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    validation_size = len(full_dataset) - train_size 
    train_dataset,validation_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validation_size] 
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    weights = [total_out_speaker_clips / count for count in speaker_clip_counts]
    weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)
    weights[-1] = 1
    weights = weights.to(device)

    classifier = Classifier(num_classes=NUM_CLASSES).to(device)
    print(next(classifier.parameters()).device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

    # disable weights when we aren't using junk
    initial_epoch_count = 0

    if USE_CHECKPOINT:
        print("INFO: Loading state from latest saved model")
        checkpoint = torch.load(MODEL_PATH)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch_count = checkpoint["epoch"]
        print(f"INFO: Beginning from epoch {initial_epoch_count}")

    wandb.init(project='automatic-speaker-recognition')
    wandb.watch(classifier)

    print('INFO: Starting training...')
    for epoch_num, epoch in enumerate(range(NUM_EPOCHS)):

        wandb.log({"epoch": initial_epoch_count + epoch_num + 1})
        wandb.log({"batch_size": BATCH_SIZE}) 
        wandb.log({"learning_rate": LEARNING_RATE})
        wandb.log({"random_seed": random_seed})

        classifier.train()
        running_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(train_loader):
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
                msg = f"INFO: [{initial_epoch_count + epoch_num + 1}, {batch_index + 1}]: train loss: {running_loss / 120}"
                print(msg)
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
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            validation_loss += loss.item()
            if batch_index % 120 == 119:
                msg = f"INFO: [{initial_epoch_count + epoch_num + 1}, {batch_index + 1}]: val loss: {validation_loss / 120}"
                print(msg)
                wandb.log({'validation_loss': validation_loss / 120})
                validation_loss = 0.0

        # only perform 'expensive' computation of heatmap, f1 every 10th epoch
        if epoch_num % 10 == 9:
            weighted_f1, micro_f1, macro_f1, all_labels, all_predicted = test_classifier(classifier, validation_loader, NUM_CLASSES)

            all_labels = np.array([int(x) for x in all_labels])
            all_predicted = np.array([int(x) for x in all_predicted])
            data = confusion_matrix(all_labels, all_predicted, normalize='true')

            ax = sns.heatmap(data)
            ax.invert_yaxis()
            fig = ax.get_figure()
            plt.legend([],[], frameon=False)
            fig.savefig(f'heatmap-{epoch_num}.png') 
            plt.clf()
            plt.cla()
            plt.close()
            wandb.save(f'heatmap-{epoch_num}.png')

            wandb.log({'validation_weighted_f1': weighted_f1})
            wandb.log({'validation_micro_f1': micro_f1})
            wandb.log({'validation_macro_f1': macro_f1})
            print(f'INFO: Weighted F1 on validation set: {weighted_f1}')
            print(f'INFO: Micro F1 on validation set: {micro_f1}')
            print(f'INFO: Macro F1 on validation set: {macro_f1}')

        torch.save(
            {
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'random_seed': random_seed,
                'in_speaker_ratio': IN_SPEAKER_RATIO,
                'train_split': TRAIN_SPLIT
            },
            MODEL_PATH,
        )

        wandb.save('model.pt')


if __name__ == "__main__":
    main(use_checkpoint=False, in_speaker_ratio=0.8, num_epochs=400, batch_size=32, random_seed=101)
