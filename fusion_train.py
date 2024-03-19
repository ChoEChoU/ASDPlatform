import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from itertools import cycle


asd_hr_train_features = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/asd_hr/feature/best_features.npy').mean(axis=(2, 3, 4))
asd_hr_train_labels = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/asd_hr/feature/best_labels.npy')

hr_td_train_features = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/hr_td/feature/best_features.npy').mean(axis=(2, 3, 4))
hr_td_train_labels = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/hr_td/feature/best_labels.npy') + 1

asd_hr_test_features = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/asd_hr/feature/best_test_asd_hr_features.npy').mean(axis=(2, 3, 4))
asd_hr_test_labels = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/asd_hr/feature/best_test_asd_hr_labels.npy')

hr_td_test_features = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/hr_td/feature/best_test_hr_td_features.npy').mean(axis=(2, 3, 4))
hr_td_test_labels = np.load('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/hr_td/feature/best_test_hr_td_labels.npy') + 1

# hr_td 테스트 데이터셋에서 TD 클래스에 해당하는 데이터만 선택
hr_td_test_features_filtered = hr_td_test_features[hr_td_test_labels == 2]
hr_td_test_labels_filtered = hr_td_test_labels[hr_td_test_labels == 2]
# hr_td_test_features_filtered = hr_td_test_features
# hr_td_test_labels_filtered = hr_td_test_labels

# asd_hr_test_features_filtered = asd_hr_test_features[asd_hr_test_labels == 0]
# asd_hr_test_labels_filtered = asd_hr_test_labels[asd_hr_test_labels == 0]

# 훈련 및 테스트 데이터셋 합치기
train_features = np.concatenate((asd_hr_train_features, hr_td_train_features), axis=0)
train_labels = np.concatenate((asd_hr_train_labels, hr_td_train_labels), axis=0)
test_features = np.concatenate((asd_hr_test_features, hr_td_test_features_filtered), axis=0)
test_labels = np.concatenate((asd_hr_test_labels, hr_td_test_labels_filtered), axis=0)
# test_features = np.concatenate((asd_hr_test_features_filtered, hr_td_test_features), axis=0)
# test_labels = np.concatenate((asd_hr_test_labels_filtered, hr_td_test_labels), axis=0)

# PyTorch 텐서로 변환
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# DataLoader 생성
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class MultiClassMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MultiClassMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # alpha 처리
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha[targets]
        else:
            at = 1.0
        
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 모델, 손실 함수, 최적화 알고리즘 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiClassMLP(input_size=train_features.shape[1], hidden_size=512, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
# num_samples = [72, 71, 44]  # 각 클래스의 샘플 수 예시
# total_samples = sum(num_samples)
# alpha = [total_samples / (len(num_samples) * x) for x in num_samples]
# criterion = FocalLoss(alpha=alpha, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# 테스트 함수
def test_model_and_evaluate(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_score.extend(outputs.cpu().numpy())
            print(y_true)
            print(y_pred)
            print(y_score)

    # 멀티클래스 문제의 경우 label_binarize를 사용하여 레이블을 이진화
    y_true_binary = label_binarize(y_true, classes=[0, 1, 2])
    y_score = np.array(y_score)

    return y_true, y_pred, y_score, y_true_binary

train_model(model, train_loader, criterion, optimizer)

# 모델 테스트 및 성능 메트릭 계산
y_true, y_pred, y_score, y_true_binary = test_model_and_evaluate(model, test_loader, device)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

# 혼동 행렬 그리기
def plot_confusion_matrix(y_true, y_pred):
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    # 클래스 레이블 정의
    class_labels = ['ASD', 'HR', 'TD']
    # 혼동 행렬 시각화
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/best_matrix.png')
    plt.close()  # 다음 그래프에 영향을 주지 않도록 그래프 창을 닫음
    print(cm)
plot_confusion_matrix(y_true, y_pred)

# 멀티클래스 AUROC 그래프 그리기
def plot_multiclass_roc(y_true, y_score, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(f'Class {i} ROC AUC: {roc_auc[i]:.4f}')  # 각 클래스별 ROC AUC 점수 출력

    # Plot all ROC curves
    plt.figure()
    colors = cycle(['blue', 'red', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC')
    plt.legend(loc="lower right")
    plt.savefig('/home/lmtna99/ASDPlatform/mmaction2/scratch/output/seventh/fold1/best_AUROC.png')
    plt.close()

def calculate_and_print_auroc(y_true, y_score, n_classes):
    # 이진화된 레이블이 필요
    y_true_binary = label_binarize(y_true, classes=range(n_classes))
    
    # 각 클래스에 대한 AUROC 점수 계산
    roc_auc = roc_auc_score(y_true_binary, y_score, multi_class='ovr')
    
    print(f'Overall AUROC: {roc_auc:.4f}')
    
    # 각 클래스별 AUROC 점수 계산
    for i in range(n_classes):
        roc_auc_i = roc_auc_score(y_true_binary[:, i], y_score[:, i])
        print(f'AUROC of class {i}: {roc_auc_i:.4f}')

calculate_and_print_auroc(y_true, y_score, n_classes=3)
plot_multiclass_roc(y_true_binary, y_score, n_classes=3)