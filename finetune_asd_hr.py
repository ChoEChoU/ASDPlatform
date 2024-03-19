import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
from temporal_transforms import TemporalRandomCrop
from mmaction.registry import MODELS
import mmengine
from mmengine.registry import init_default_scope
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Initialize Weights & Biases
wandb.init(project="mmaction2-tools")

# Configuration
num_classes = 2  # Set the number of classes in your dataset
num_epochs = 100  # Set the number of training epochs
batch_size = 16  # Set the batch size
learning_rate = 1e-6  # Learning rate for the optimizer
T_max = num_epochs
config_path = '/home/lmtna99/ASDPlatform/mmaction2/configs/recognition/i3d/custom/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py'
model_path = '/home/lmtna99/ASDPlatform/I3D_scratch/pretrain_second_best_model.pth'
checkpoint_dir = "/home/lmtna99/ASDPlatform/mmaction2/scratch/output/eighth/fold5/asd_hr"  # Specify your checkpoint directory
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
    # def load_frames(self, video_path, num_frames):
    #     frame_paths = [os.path.join(video_path, f'img_{i:05d}.jpg') for i in range(num_frames)]

    #     def load_frame(path):
    #         return Image.open(path)

    #     with ThreadPoolExecutor(max_workers=10) as executor:
    #         frames = list(executor.map(load_frame, frame_paths))

    #     return frames
    
def load_model(model_path, model):
    checkpoint = torch.load(model_path)
    # print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Define Transform
spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

temporal_transform = TemporalRandomCrop(size=64)

# Load Datasets
train_dataset = MyDataset('/mnt/2021_NIA_data/projects/interaction_video/output/crop_data/IF2004/fusion/ASD_HR_fold_5_train.txt', transform=spatial_transform, temporal_transform=temporal_transform, is_train=True)
val_dataset = MyDataset('/mnt/2021_NIA_data/projects/interaction_video/output/crop_data/IF2004/fusion/ASD_HR_fold_5_validation.txt', transform=spatial_transform, temporal_transform=temporal_transform, is_train=False)
test_dataset = MyDataset('/mnt/2021_NIA_data/projects/interaction_video/output/crop_data/IF2004/fusion/ASD_HR_fold_5_test.txt', transform=spatial_transform, temporal_transform=temporal_transform, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

# Initialize the I3D model
i3d_model = MODELS.build(config.model)
i3d_model = load_model(model_path, i3d_model)
i3d_model.to(device)

def freeze_except_specific_layers(model, except_layer_paths):
    # 모든 파라미터를 freeze
    for param in model.parameters():
        param.requires_grad = False

    # 여러 레이어의 경로를 리스트로 받아 각각에 대해 unfreeze 수행
    for layer_path in except_layer_paths:
        modules = layer_path.split('.')
        curr_module = model
        for m in modules:
            if hasattr(curr_module, m):
                curr_module = getattr(curr_module, m)
            else:
                print(f"Module or attribute '{m}' not found in the current module.")
                return  # 해당 모듈을 찾지 못한 경우 함수 종료

        # 찾은 모듈의 파라미터를 unfreeze
        for param in curr_module.parameters():
            param.requires_grad = True

    print(f"All layers except {except_layer_paths} have been frozen.")

# freeze_except_specific_layers(i3d_model, ['backbone.layer2','backbone.layer3','backbone.layer4', 'cls_head'])

def print_frozen_layers(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Layer '{name}' is frozen.")

# 모델의 freeze된 레이어 확인
print_frozen_layers(i3d_model)

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

# Initialize the cls_head
# cls_head = i3d_model.cls_head
# cls_head.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(i3d_model.parameters(), lr=learning_rate, weight_decay=0.1)

# Cosine Annealing Scheduler
scheduler = CosineAnnealingLR(optimizer, T_max=T_max)

# Training Loop
best_accuracy = 0.0
for epoch in range(num_epochs):
    i3d_model.train()
    running_loss = 0.0
    total_accuracy = 0
    total_batches = 0
    set_feature_extraction_status(True)
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(0)

        optimizer.zero_grad()
        outputs = i3d_model(inputs)
        outputs = i3d_model.cls_head(outputs)
        if outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / inputs.size(1)
        
        total_accuracy += accuracy

        # Log batch accuracy to wandb
        wandb.log({"Batch Accuracy": accuracy})
        
        running_loss += loss.item()

        if  total_batches % 10 == 0:  # Adjust logging interval as needed
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{total_batches}/{len(train_loader)}], Loss: {loss.item()}')
            wandb.log({"batch_loss": loss.item()})
            
        total_batches += 1
        
        labels_list.extend(labels.cpu().numpy())
    
    scheduler.step()
    
    avg_loss = running_loss / len(train_loader)
    avg_epoch_accuracy = total_accuracy / total_batches
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0], "Mean Accuracy": avg_epoch_accuracy})

    # Validation after each epoch
    i3d_model.eval()
    i3d_model.cls_head.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    set_feature_extraction_status(False)
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', unit='batch'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(0)
            outputs = i3d_model(inputs)
            outputs = i3d_model.cls_head(outputs)
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += inputs.size(1)
            correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    
    wandb.log({"val_loss": avg_val_loss, "val_accuracy": val_accuracy, "epoch": epoch})
    
    if epoch % 10 == 0:
        check_epoch = epoch + 1
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{check_epoch}.pth")
        torch.save(i3d_model.state_dict(), checkpoint_path)

    # Save the best model
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_epoch = epoch
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(i3d_model.state_dict(), best_model_path)
        print(f"Best model found at epoch {best_epoch} with accuracy {best_accuracy:.2f}.")
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.array(labels_list)
        
        np_features_save_path = os.path.join(checkpoint_dir, "best_features.npy")
        np.save(np_features_save_path, features_array)
        np_labels_save_path = os.path.join(checkpoint_dir, "best_labels.npy")
        np.save(np_labels_save_path, labels_array)
        print(features_array.shape, labels_array.shape)
    
    gc.collect()
    if epoch != num_epochs - 1:
        del features_list[:]
        del labels_list[:]

# Save the trained model
last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
torch.save(i3d_model.state_dict(), last_model_path)
hook.remove()

features_array = np.concatenate(features_list, axis=0)
labels_array = np.array(labels_list)

np_features_save_path = os.path.join(checkpoint_dir, "last_features.npy")
np.save(np_features_save_path, features_array)
np_labels_save_path = os.path.join(checkpoint_dir, "last_labels.npy")
np.save(np_labels_save_path, labels_array)

def evaluate_performance(labels, predictions, scores):
    # Confusion Matrix 계산
    cm = confusion_matrix(labels, predictions)
    # Precision, Recall, F1-Score 계산
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    # AUROC 계산
    auroc = roc_auc_score(labels, scores)
    
    accuracy = np.mean(np.array(labels) == np.array(predictions))
    
    return cm, precision, recall, f1, auroc, accuracy

def plot_auroc(labels, scores, save_path):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()  # 그래프 창을 닫아 다음 그래프에 영향을 주지 않도록 함
    
def plot_confusion_matrix(cm, class_names, save_path):
    """
    cm: Confusion matrix array
    class_names: 클래스 이름 목록
    save_path: 저장할 파일 경로
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

all_labels = []
all_predictions = []
all_scores = []

i3d_model.eval()
i3d_model.cls_head.eval()
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc='Test', unit='batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(0)
        outputs = i3d_model(inputs)
        outputs = i3d_model.cls_head(outputs)

        # 점수 계산 (softmax 활성화 함수 적용)
        scores = torch.softmax(outputs, dim=1)[:, 1]  # Positive class에 대한 확률
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())
        
cm, precision, recall, f1, auroc, accuracy = evaluate_performance(all_labels, all_predictions, all_scores)

# 평가 지표 출력
print(f'Confusion Matrix:\n{cm}')
print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUROC: {auroc:.4f}, Accuracy: {accuracy:.4f}')

# 클래스 이름 설정
class_names = ['ASD', 'HR']

# Confusion Matrix 저장 경로 설정
cm_save_path = os.path.join(checkpoint_dir, 'last_confusion_matrix.png')

# Confusion Matrix 그리기 및 저장
plot_confusion_matrix(cm, class_names, cm_save_path)

auroc_plot_path = os.path.join(checkpoint_dir, 'last_auroc_curve.png')

# AUROC 그래프 그리기 및 저장
plot_auroc(all_labels, all_scores, auroc_plot_path)