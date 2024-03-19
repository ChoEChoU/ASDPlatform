import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms
from temporal_transforms import TemporalRandomCrop
from mmaction.registry import MODELS
import mmengine
from mmengine.registry import init_default_scope
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random

def set_seed(seed=42):
    """모든 라이브러리의 무작위성 제어를 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU 사용시
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

config_path = '/home/lmtna99/ASDPlatform/mmaction2/configs/recognition/i3d/custom/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py'
model_path = '/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/asd_hr/model/best_model.pth'
checkpoint_dir = "/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/asd_hr"  # Specify your checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)
num_classes = 2

# GPU configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# config setting
config = mmengine.Config.fromfile(config_path)
init_default_scope(config.get('default_scope', 'mmaction'))
config.model.backbone.pretrained = None

# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, annotation_file, transform=None, temporal_transform=None, is_train=True):
        self.annotations = [line.strip().split() for line in open(annotation_file, 'r')]
        self.transform = transform
        self.temporal_transform = temporal_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        video_path, num_frames, label = self.annotations[idx]
        frames = self.load_frames(video_path, int(num_frames))

        if self.temporal_transform:
            frames = self.temporal_transform(frames)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = frames.permute(1, 0, 2, 3)  # Change to [C, D, H, W]
        label = torch.tensor(int(label))
        
        return frames, label

    def load_frames(self, video_path, num_frames):
        frames = []
        for i in range(num_frames):
            frame_filename = f'img_{i:05d}.jpg'
            frame_path = os.path.join(video_path, frame_filename)
            frame = Image.open(frame_path)
            frames.append(frame)
        return frames
    
def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    return model

# Define Transform
spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

temporal_transform = TemporalRandomCrop(size=64)

test_dataset = MyDataset('/mnt/2021_NIA_data/projects/interaction_video/output/additional_240108/crop_interaction/IF2004_crop/fusion/ASD_HR_fold_5_test.txt', transform=spatial_transform, temporal_transform=temporal_transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

# Initialize the I3D model
i3d_model = MODELS.build(config.model)
i3d_model = load_model(model_path, i3d_model)
i3d_model.to(device)

# Initialize the cls_head
# cls_head = i3d_model.cls_head
# cls_head.to(device)

# Perform predictions
y_true = []
y_pred = []
y_scores = []
i3d_model.eval()
i3d_model.cls_head.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Test', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = i3d_model(inputs)
        outputs = i3d_model.cls_head(outputs)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        y_scores.extend(probabilities.cpu().numpy())
        
print(y_true, y_pred)
        
# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Plot and save the confusion matrix
class_names = ['ASD', 'HR']

plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion Matrix')
plt.savefig('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/asd_hr/best_confusion_matrix.png')
plt.close()

fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(y_true) == i, np.array(y_scores)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f'Class {i} ROC AUC: {roc_auc[i]:.4f}')

# Plot ROC Curve for each class
plt.figure(figsize=(8, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_curve_path = '/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/asd_hr/best_roc_curve.png'
plt.savefig(roc_curve_path)
plt.close()