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

config_path = '/home/lmtna99/ASDPlatform/mmaction2/configs/recognition/i3d/custom/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py'
model_path = '/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/hr_td/model/best_model.pth'
checkpoint_dir = "/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold5/hr_td/feature"  # Specify your checkpoint directory
os.makedirs(checkpoint_dir, exist_ok=True)

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

test_dataset = MyDataset('/mnt/2021_NIA_data/projects/interaction_video/output/additional_240108/crop_interaction/IF2004_crop/fusion/HR_TD_fold_5_test.txt', transform=spatial_transform, temporal_transform=temporal_transform, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

# Initialize the I3D model
i3d_model = MODELS.build(config.model)
i3d_model = load_model(model_path, i3d_model)
i3d_model.to(device)

extract_features = False
features_list = []
labels_list = []
def hook_fn(module, input, output):
    if extract_features:
        features_list.append(output.cpu().detach().numpy())

hook = i3d_model.backbone.layer4[2].register_forward_hook(hook_fn)

def set_feature_extraction_status(status):
    global extract_features
    extract_features = status

set_feature_extraction_status(True)
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Test', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = i3d_model(inputs)
        labels_list.extend(labels.cpu().numpy())
        
features_array = np.concatenate(features_list, axis=0)
labels_array = np.array(labels_list)

np_features_save_path = os.path.join(checkpoint_dir, "best_test_hr_td_features.npy")
np.save(np_features_save_path, features_array)
np_labels_save_path = os.path.join(checkpoint_dir, "best_test_hr_td_labels.npy")
np.save(np_labels_save_path, labels_array)