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
from pydub import AudioSegment
from sklearn.metrics import f1_score
from tensorflow.python.keras import backend as K
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel

# import wandb
# wandb.init(project="automatic-speaker-recognition")


class ClassifierDataset(Dataset):
    """Load numpy files from directory structure where each numpy file represents
    the extracted features from the pre-trained model"""

    def __init__(self, directory):
        outputs = []
        labels = []

        speakers = [f for f in os.listdir(directory) if f != ".DS_Store"]
        for i, speaker in enumerate(speakers):
            for clip in os.listdir(f"{directory}/{speaker}"):
                if "npy" not in clip:
                    continue

                output = np.load(f"{directory}/{speaker}/{clip}")

                outputs.append(output)
                labels.append(i)

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
def test_classifier(classifier, classifier_testing_loader, count):
    #class_correct = [0] * count
    #class_total = [0] * count

    # used to calculate global f1
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in classifier_testing_loader:
            images, labels = data
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            #c = (predicted == labels).squeeze()

            all_labels += labels
            all_predicted += predicted

    f1 = f1_score(all_labels, all_predicted, average="weighted")

    return f1


def main():
    print('INFO: Starting training...')
    np.random.seed(1234)
    random.seed(1234)

    AUDIO_PATH = "audio"
    SOURCE_DIR = "accents_features"
    MODEL_PATH = "model.pt"
    USE_CHECKPOINT = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 400
    TRAIN_SPLIT = 0.2
    LEARNING_RATE = 0.003

    dataset_dir = f"{AUDIO_PATH}/{SOURCE_DIR}"
    full_dataset = ClassifierDataset(dataset_dir)

    CLASSES = [f for f in os.listdir(dataset_dir) if f != ".DS_Store"]

    train_size = int(TRAIN_SPLIT * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    classifier = Classifier(num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    initial_epoch_count = 0

    if USE_CHECKPOINT:
        print("INFO: Loading state from latest saved model")
        checkpoint = torch.load(MODEL_PATH)
        classifier.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch_count = checkpoint["epoch"]
        print(f"INFO: Beginning from epoch {initial_epoch_count}")

    # weights = [subject_weight] * num_classes
    # weights[-1] = junk_weight
    # weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

    # criterion = nn.CrossEntropyLoss(weight=weights, reduction='mean')
    # disable weights when we aren't using junk

    # wandb.watch(classifier)

    for epoch_num, epoch in enumerate(range(NUM_EPOCHS)):

        # wandb.log({"epoch": initial_epoch_count + epoch_num + 1})

        classifier.train()
        running_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_index % 120 == 119:
                msg = f"INFO: [{initial_epoch_count + epoch_num + 1}, {batch_index + 1}]: loss: {running_loss / 120}"
                print(msg)
                # wandb.log({'train_loss': running_loss / 120})
                running_loss = 0.0

        classifier.eval()
        validation_loss = 0.0
        for batch_index, (inputs, labels) in enumerate(test_loader):
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            validation_loss += loss.item()
            if batch_index % 120 == 119:
                msg = f"INFO: [{initial_epoch_count + epoch_num + 1}, {batch_index + 1}]: loss: {validation_loss / 120}"
                print(msg)
                # wandb.log({'validation_loss': validation_loss / 120})

                validation_loss = 0.0

        torch.save(
            {
                "epoch": initial_epoch_count + epoch_num,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            MODEL_PATH,
        )


"""
classifier = Classifier(num_classes=len(classes))
checkpoint = torch.load(MODEL_PATH)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()
f1 = test_classifier(classifier, test_loader, len(classes), output_stats=True)
"""

if __name__ == "__main__":
    main()
