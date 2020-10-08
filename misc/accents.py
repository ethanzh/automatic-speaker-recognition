import logging
import os
import random
import sys
from functools import partial
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from scipy.stats import entropy
from sklearn.metrics import f1_score
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tornado import gen

from audio import read_mfcc
from batcher import sample_from_mfcc
from constants import NUM_FRAMES, SAMPLE_RATE
from conv_models import DeepSpeakerModel

np.random.seed(1234)
random.seed(1234)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)


class ClassifierDataset(Dataset):
    """Load numpy files from directory structure where each numpy file represents
    the extracted features from the pre-trained model"""

    def __init__(self, positive_labels, negative_labels, num_shots=2, train=False):
        outputs = []
        labels = []

        positive_labels = [f for f in positive_labels if f != ".DS_Store"]

        for i, speaker_dir in enumerate(positive_labels):
            relative_path = f"mfcc/split/Accents/{speaker_dir}"
            samples = os.listdir(relative_path)

            sample_range = range(num_shots) if train else range(num_shots, len(samples))
            for j in sample_range:
                output = np.load(f"{relative_path}/Accents_{j}.npy")[0]
                outputs.append(output)
                labels.append(i)

        negative_index = labels[-1] + 1
        negative_labels = [f for f in negative_labels if f != ".DS_Store"]

        for speaker_dir in negative_labels:
            relative_path = f"mfcc/split/Accents/{speaker_dir}"
            samples = os.listdir(relative_path)

            sample_range = range(num_shots) if train else range(num_shots)
            for j in sample_range:
                output = np.load(f"{relative_path}/Accents_{j}.npy")[0]
                outputs.append(output)
                labels.append(negative_index)

        self.outputs = np.array(outputs)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return torch.tensor(self.outputs[idx]), self.labels[idx]


def generate_dataloaders(count=None, junk_count=None, min_clips=3, num_shots=1):
    accents_mfcc = [f for f in os.listdir("mfcc/split/Accents") if f != ".DS_Store"]

    filtered_accents_mfcc = [
        s
        for s in accents_mfcc
        if len(os.listdir(f"mfcc/split/Accents/{s}")) >= min_clips
    ]

    random.shuffle(filtered_accents_mfcc)

    num_segments = 2 #3

    split_data = np.array_split(filtered_accents_mfcc, num_segments)

    # needed to know how many output neurons for classifier
    classifier_classes = split_data[0]
    #validation_set = split_data[1]
    #val_set_1, val_set_2, val_set_3, val_junk = np.array_split(validation_set, 4)
    #validation_set = np.concatenate((val_set_1, val_set_2, val_set_3))
    #junk_data = split_data[2]
    junk_data = split_data[1]

    junk_data_1, junk_data_2 = np.array_split(junk_data, 2)

    # use a subset of the data based on count
    if count:
        classifier_classes = classifier_classes[:count]

    if junk_count:
        junk_data_1 = junk_data_1[:junk_count]
        # junk_data_2 = junk_data_2[:junk_count]

    batch_size = 16

    if count and len(classifier_classes) < count:
        print(
            f"WARNING: Count is higher than number of classes available: {len(classifier_classes)}"
        )

    classifier_training_dataset = ClassifierDataset(
        classifier_classes, junk_data_1, num_shots=num_shots, train=True
    )
    classifier_testing_dataset = ClassifierDataset(
        classifier_classes, junk_data_2, num_shots=num_shots, train=False
    )

    #classifier_validation_dataset = ClassifierDataset(
        #validation_set, val_junk, num_shots=num_shots, train=True
    #)

    classifier_training_loader = DataLoader(
        classifier_training_dataset, batch_size=batch_size
    )
    classifier_testing_loader = DataLoader(
        classifier_testing_dataset, batch_size=batch_size
    )
    #classifier_validation_loader = DataLoader(
        #classifier_validation_dataset, batch_size=batch_size
    #)

    return (
        classifier_training_loader,
        classifier_testing_loader,
        #classifier_validation_loader,
    )


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


def train_classifier(
    classifier_training_loader,
    classifier_validation_loader,
    subject_weight,
    junk_weight,
    num_classes,
    num_epochs=1000,
    lr=0.003,
    returns_as_function=False
):
    classifier = Classifier(num_classes=num_classes)

    weights = [subject_weight] * num_classes
    weights[-1] = junk_weight
    weights = torch.from_numpy(np.array(weights)).type(torch.FloatTensor)

    criterion = nn.CrossEntropyLoss(weight=weights, reduction="mean")
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    training_losses = []

    for epoch_num, epoch in enumerate(range(num_epochs)):
        train_loss = 0.0
        validation_loss = 0.0

        classifier.train()
        for batch_index, (inputs, labels) in enumerate(classifier_training_loader):
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            train_loss += loss.item() * inputs.size(0)
            optimizer.step()

        classifier.eval()
        for batch_index, (inputs, labels) in enumerate(
            classifier_validation_loader
        ):
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            validation_loss += loss.item() * inputs.size(0)
            optimizer.step()

        train_loss = train_loss / len(classifier_training_loader)
        training_losses.append(train_loss)
        validation_loss = validation_loss / len(classifier_validation_loader)

        new_data = {
            "epochs": [epoch_num],
            "trainlosses": [train_loss],
            "vallosses": [validation_loss],
        }
        doc.add_next_tick_callback(partial(update, new_data))

    return classifier, training_losses


def test_classifier(classifier, classifier_testing_loader, count, output_stats=False):
    class_correct = [0] * count
    class_total = [0] * count

    # used to calculate global f1
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for data in classifier_testing_loader:
            images, labels = data
            outputs = classifier(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            all_labels += labels
            all_predicted += predicted

            """
            for i, (output, label) in enumerate(zip(outputs, labels)):
                class_total[label] += 1
                class_correct[label] += c[i].item()
            """

    f1 = f1_score(all_labels, all_predicted, average="weighted")

    if output_stats:
        print(f"f1: {f1}")

    return f1


def pipeline(
    count, subject_weight, junk_weight, junk_count=30, min_clips=3, num_shots=1
):
    (
        classifier_training_loader,
        classifier_testing_loader,
        #classifier_validation_loader,
    ) = generate_dataloaders(
        count=count, junk_count=junk_count, min_clips=min_clips, num_shots=num_shots
    )
    classifier, training_losses = train_classifier(
        classifier_training_loader,
        classifier_testing_loader,
        subject_weight,
        junk_weight,
        count + 1,
    )

    # find optimal thresholds
    f1 = test_classifier(
        classifier, classifier_testing_loader, count + 1, output_stats=False
    )

    return f1, training_losses


from bokeh.io import output_file, show


def test_model():
    output_file("index.html")
    highest_f1 = 0
    # (subject count, num clips, junk count)
    highest_values = (None, None, None)

    NUM_SHOTS = 2

    min_num_clips = 3

    min_subject_count = 100 
    max_subject_count = 299

    max_f1_per_subject_count = {}
    values_per_subject_count = {}


    for subject_count in range(min_subject_count, max_subject_count, 50):
        for junk_count in range(int(subject_count / 2), int(subject_count * 2), 50):
            subject_weight = junk_count / min_num_clips
            junk_weight = 1

            f1, training_losses = pipeline(subject_count, subject_weight=subject_weight, junk_weight=junk_weight, junk_count=junk_count, min_clips=min_num_clips, num_shots=NUM_SHOTS)

            source = ColumnDataSource(data={"epochs": range(1000), "trainlosses": training_losses})
            plot = figure(title=f"subjects: {subject_count}, junk: {junk_count}, f1: {f1}")
            plot.line(
                x="epochs",
                y="trainlosses",
                color="green",
                alpha=0.8,
                legend_label="Train loss",
                line_width=2,
                source=source,
            )

            rounded_f1 = round(f1, 2)
            print(f'f1 (subjects: {subject_count}, junk: {junk_count}): {rounded_f1}')

            if f1 > max_f1_per_subject_count.get(subject_count, 0):
                values = (subject_count, min_num_clips, junk_count)
                print(f'***NEW HIGH*** f1 (subjects: {subject_count}, junk: {junk_count}): {rounded_f1}')
                max_f1_per_subject_count[subject_count] = f1
                values_per_subject_count[subject_count] = values
                show(plot)

test_model()
"""
doc = curdoc()

@gen.coroutine
def update(new_data):
    source.stream(new_data)

fun = test_model(doc)

#thread = Thread(target=fun)
#thread.start()

plot.line(
    x="epochs",
    y="vallosses",
    color="red",
    alpha=0.8,
    legend_label="Val loss",
    line_width=2,
    source=source,
)
"""
