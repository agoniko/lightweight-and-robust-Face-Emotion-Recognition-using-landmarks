import os
import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from mediapipe.framework.formats import landmark_pb2
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.linalg import lstsq

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def draw_landmarks(image, landmark, already_normalized=False):

    if not already_normalized:
        normalized_landmarks = [
            landmark_pb2.NormalizedLandmark(x=val[0], y=val[1], z=val[2])
            for val in landmark
        ]
        l_list = landmark_pb2.NormalizedLandmarkList(landmark=normalized_landmarks)
    else:
        l_list = landmark_pb2.NormalizedLandmarkList(landmark=landmark)

    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=l_list,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    return image


def draw_landmarks_colored(image, landmark, already_normalized=False, selected_indices=None, selected_color=(255, 0, 0), default_color=(0, 0, 255)):
    # Prepare normalized landmarks
    if not already_normalized:
        normalized_landmarks = [
            landmark_pb2.NormalizedLandmark(x=val[0], y=val[1], z=val[2])
            for val in landmark
        ]
        l_list = landmark_pb2.NormalizedLandmarkList(landmark=normalized_landmarks)
    else:
        l_list = landmark_pb2.NormalizedLandmarkList(landmark=landmark)

    # Drawing specification for default landmarks
    default_drawing_spec = mp_drawing.DrawingSpec(color=default_color, thickness=1, circle_radius=1)

    # Drawing specification for selected landmarks
    selected_drawing_spec = mp_drawing.DrawingSpec(color=selected_color, thickness=2, circle_radius=2)

    # Draw landmarks
    for idx, lm in enumerate(l_list.landmark):
        # Check if this index is in the selected indices
        if selected_indices is not None and idx in selected_indices:
            # Use the selected drawing spec
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmark_pb2.NormalizedLandmarkList(landmark=[lm]),
                landmark_drawing_spec=selected_drawing_spec,
                connection_drawing_spec=None,
            )
        else:
            # Use the default drawing spec
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=landmark_pb2.NormalizedLandmarkList(landmark=[lm]),
                landmark_drawing_spec=default_drawing_spec,
                connection_drawing_spec=None,
            )

    # Draw connections using default style
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=l_list,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )

    return image


def fast_draw(image, landmarks):
    assert type(landmarks) == torch.Tensor or type(landmarks) == np.ndarray
    h, w = image.shape[:2]
    if landmarks.shape[1] == 3:
        landmarks = landmarks.permute(1, 0)

    assert landmarks.shape[0] == 3
    landmarks[0] *= w
    landmarks[1] *= h
    for x, y in zip(landmarks[0], landmarks[1]):
        cv2.circle(image, (int(x), int(y)), 2, (90, 90, 90), -1)

    return image


class EmotionDataset(Dataset):
    def __init__(self, dataset, label2id, normalization_func, indices=None):
        self.dataset = dataset
        self.landmarks = [x["landmarks"] for x in self.dataset]
        self.landmarks = torch.tensor(
            [normalization_func(np.array(x), indices) for x in self.landmarks],
            dtype=torch.float32,
        )
        self.labels = torch.tensor(
            [label2id[x["emotion"]] for x in self.dataset], dtype=torch.long
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.landmarks[idx].permute(1, 0), self.labels[idx]


class Net(nn.Module):
    def __init__(self, out_size, dropout_p=0.1):
        super(Net, self).__init__()
        self.back = nn.Sequential(
            nn.Conv1d(3, 64, 8),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 6),
            nn.MaxPool1d(2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 4),
            nn.MaxPool1d(2),
            # nn.Dropout1d(0.0),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3),
            nn.MaxPool1d(2),
            # nn.Dropout1d(0.0),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 3, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_size),
        )

        for conv_layer in self.back:
            if isinstance(conv_layer, nn.Conv1d):
                nn.init.kaiming_normal_(conv_layer.weight)
        for fc_layer in self.fc:
            if isinstance(fc_layer, nn.Linear):
                nn.init.kaiming_normal_(fc_layer.weight)

    def forward(self, x):
        x = self.back(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Net2D(nn.Module):
    def __init__(self, out_size):
        super(Net2D, self).__init__()
        self.back = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 4, 32), nn.Dropout(0.3), nn.ReLU(), nn.Linear(32, out_size)
        )

        # Initialize weights using Kaiming initialization
        for conv_layer in self.back:
            if isinstance(conv_layer, nn.Conv2d):
                nn.init.kaiming_normal_(conv_layer.weight)
        for fc_layer in self.fc:
            if isinstance(fc_layer, nn.Linear):
                nn.init.kaiming_normal_(fc_layer.weight)

    def forward(self, x):
        x = self.back(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def minmax_scale(data, indices=None):
    assert data.shape[1] == 3
    if indices is not None:
        data = data[indices]
    minval = np.min(data, axis=0)
    maxval = np.max(data, axis=0)
    return (data - minval) / (maxval - minval)


def std_norm(landmarks, mean=0.5):
    assert landmarks.shape[1] == 3
    std = np.std(landmarks, axis=0)
    mean_values = np.mean(landmarks, axis=0)

    landmarks = (landmarks - mean_values) / std
    landmarks = landmarks * mean + mean
    return landmarks


def y_scale(landmarks):
    # Extract x, y, z coordinates
    x_coords = landmarks[:, 0]
    y_coords = landmarks[:, 1]
    z_coords = landmarks[:, 2]

    # Find the min and max of y coordinates
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Scale y to range [0, 1]
    y_scaled = (y_coords - y_min) / (y_max - y_min)

    # Calculate the scaling factor based on y scaling
    scale_factor = 1 / (y_max - y_min)

    # Apply the same scaling to x and z coordinates
    x_scaled = (x_coords - np.mean(x_coords)) * scale_factor + 0.5
    z_scaled = (z_coords - np.mean(z_coords)) * scale_factor + 0.5

    # Combine the scaled coordinates back into the landmarks array
    landmarks_scaled = np.stack((x_scaled, y_scaled, z_scaled), axis=-1)

    return landmarks_scaled


#OLD normalization function that used face geometry, final normalization function uses canonical face (at the bottom)
def normalize_face(landmarks, indices=None, minmax=True, std=False):
    assert landmarks.shape == (478, 3)
    assert not (minmax and std)
    landmarks = landmarks - np.mean(landmarks, axis=0)
    forehead = landmarks[10]
    chin = landmarks[152]
    left = landmarks[468]
    right = landmarks[473]

    Y_axis = forehead - chin
    X_axis = left - right

    # Gram-Schmidt orthogonalization
    Y_axis /= np.linalg.norm(Y_axis)
    X_axis -= np.dot(X_axis, Y_axis) * Y_axis
    X_axis /= np.linalg.norm(X_axis)

    # Make Z_axis orthogonal to both Y_axis and X_axis (using cross product)
    Z_axis = np.cross(Y_axis, X_axis)
    Z_axis /= np.linalg.norm(Z_axis)

    R = np.vstack((X_axis, -Y_axis, Z_axis)).T
    points = np.dot(landmarks, R)

    if indices is not None:
        points = points[indices]

    if minmax:
        points = minmax_scale(points)
    elif std:
        points = std_norm(points)

    return points

#copied indices from mediapipe repository for selected landmark subset
def get_indices_subset():
    # Connections

    LIPS_CONNECTIONS = [
        [61, 146],
        [146, 91],
        [91, 181],
        [181, 84],
        [84, 17],
        [17, 314],
        [314, 405],
        [405, 321],
        [321, 375],
        [375, 291],
        [61, 185],
        [185, 40],
        [40, 39],
        [39, 37],
        [37, 0],
        [0, 267],
        [267, 269],
        [269, 270],
        [270, 409],
        [409, 291],
        [78, 95],
        [95, 88],
        [88, 178],
        [178, 87],
        [87, 14],
        [14, 317],
        [317, 402],
        [402, 318],
        [318, 324],
        [324, 308],
        [78, 191],
        [191, 80],
        [80, 81],
        [81, 82],
        [82, 13],
        [13, 312],
        [312, 311],
        [311, 310],
        [310, 415],
        [415, 308],
    ]

    LEFT_EYE_CONNECTIONS = [
        [263, 249],
        [249, 390],
        [390, 373],
        [373, 374],
        [374, 380],
        [380, 381],
        [381, 382],
        [382, 362],
        [263, 466],
        [466, 388],
        [388, 387],
        [387, 386],
        [386, 385],
        [385, 384],
        [384, 398],
        [398, 362],
    ]

    LEFT_EYEBROW_CONNECTIONS = [
        [276, 283],
        [283, 282],
        [282, 295],
        [295, 285],
        [300, 293],
        [293, 334],
        [334, 296],
        [296, 336],
    ]

    LEFT_IRIS_CONNECTIONS = [
        [474, 475],
        [475, 476],
        [476, 477],
        [477, 474],
    ]

    RIGHT_EYE_CONNECTIONS = [
        [33, 7],
        [7, 163],
        [163, 144],
        [144, 145],
        [145, 153],
        [153, 154],
        [154, 155],
        [155, 133],
        [33, 246],
        [246, 161],
        [161, 160],
        [160, 159],
        [159, 158],
        [158, 157],
        [157, 173],
        [173, 133],
    ]

    RIGHT_EYEBROW_CONNECTIONS = [
        [46, 53],
        [53, 52],
        [52, 65],
        [65, 55],
        [70, 63],
        [63, 105],
        [105, 66],
        [66, 107],
    ]
    RIGHT_IRIS_CONNECTIONS = [
        [469, 470],
        [470, 471],
        [471, 472],
        [472, 469],
    ]

    indices = set(
        [
            x
            for sublist in LIPS_CONNECTIONS
            + LEFT_EYE_CONNECTIONS
            + LEFT_EYEBROW_CONNECTIONS
            + LEFT_IRIS_CONNECTIONS
            + RIGHT_EYE_CONNECTIONS
            + RIGHT_EYEBROW_CONNECTIONS
            + RIGHT_IRIS_CONNECTIONS
            for x in sublist
        ]
    )

    indices = np.array(list(indices))
    return indices

#training function
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    n_epochs: int,
    device: torch.device,
    save_path: str = "best_model.pth",
    logdir: str = "logs",
    patience: int = 15,
):
    global TRAIN_LOSSES, VAL_LOSSES

    writer = SummaryWriter(log_dir=logdir)
    model = model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = np.inf

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        train_loss = 0
        val_loss = 0
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        TRAIN_LOSSES = train_losses
        VAL_LOSSES = val_losses

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        scalar_dict = {
            "train": train_loss,
            "val": val_loss,
            "best_val": best_val_loss,
        }
        writer.add_scalars("Losses", scalar_dict, epoch)

        pbar.set_description(
            f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Best val loss: {best_val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if scheduler is not None:
            scheduler.step()

        if epoch > patience and min(val_losses[-patience:]) > best_val_loss:
            break

    writer.close()
    writer.flush()

    return train_losses, val_losses


def evaluate(model, test_loader, device, average: str = "macro", cm_norm: str = "true"):
    """
    average: str in ["micro", "macro", "weighted"]
    cm_norm: str in ["true", "pred", "all"]
    """
    assert average in ["micro", "macro", "weighted"]
    assert cm_norm in ["true", "pred", "all", None]
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(dim=1).cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred, normalize=cm_norm)

    return float(acc), float(f1), float(precision), float(recall), cm.tolist()

canonical_face = np.array(json.load(open("canonical_face.json")))
subset_indices = np.array([5, 33, 133, 263, 362, 61, 291])


def estimate_affine(face_landmarks, canonical_landmarks):
    # Add a column of ones to the landmarks for the affine transformation matrix
    ones = np.ones((face_landmarks.shape[0], 1))

    # Concatenate ones for the affine equation A * X = B
    face_landmarks_h = np.hstack([face_landmarks, ones])  # Homogeneous coordinates

    # Solve for the affine matrix using least squares
    affine_matrix, _, _, _ = lstsq(face_landmarks_h, canonical_landmarks)

    return affine_matrix.T  # Return the transposed matrix for transformation


def normalize_affine(face_landmarks, indices = None):
    R = estimate_affine(face_landmarks[subset_indices], canonical_face[subset_indices])
    affine_lands = np.dot(
        R, np.hstack([face_landmarks, np.ones((face_landmarks.shape[0], 1))]).T
    ).T

    if indices is not None:
        affine_lands = affine_lands[indices]
    affine_lands = minmax_scale(affine_lands)
    return affine_lands


