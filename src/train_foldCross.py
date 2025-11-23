"""
BP-SVM采煤机故障诊断系统 - Python实现 (10折交叉验证)
包括：数据加载、10折交叉验证、BP特征提取、SVM分类、结果评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score)
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
np.random.seed(42)
torch.manual_seed(42)

print('=' * 70)
print('BP-SVM采煤机故障诊断系统 (Python版 - 10折交叉验证)')
print('=' * 70)
print()

# ==================== BP神经网络定义 ====================
class BPNetwork(nn.Module):
    """BP神经网络用于特征提取"""
    def __init__(self, input_size=6, hidden_size=12, output_size=6):
        super(BPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x

def train_bp_network(X_train, y_train, num_epochs=1000, verbose=False):
    """训练BP神经网络"""
    # 使用PCA结果作为训练目标
    pca = PCA(n_components=6)
    target_features = pca.fit_transform(X_train)
    
    # 创建网络
    bp_net = BPNetwork(6, 12, 6)
    
    # 转换为张量
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(target_features)
    
    # 创建数据加载器
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 定义损失和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bp_net.parameters(), lr=0.01)
    
    # 训练
    bp_net.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = bp_net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if verbose and (epoch + 1) % 200 == 0:
            print(f'      Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    
    bp_net.eval()
    return bp_net

# ==================== 第一部分：数据加载与合并 ====================
print('步骤1: 加载并合并数据集...')

# 读取训练集和测试集
train_data = pd.read_csv('../datasets/train_data.txt', sep='\t')
test_data = pd.read_csv('../datasets/test_data.txt', sep='\t')

# 合并为总数据集
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 分离特征和标签
X_all = all_data.iloc[:, :6].values
y_all = all_data.iloc[:, 6].values

print(f'  总样本数: {len(X_all)}')
print(f'  特征维度: {X_all.shape[1]}')
print(f'  故障类型: {len(np.unique(y_all))} 类')
print()

# ==================== 第二部分：10折交叉验证设置 ====================
print('步骤2: 设置10折交叉验证...')

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

print(f'  交叉验证折数: {k_folds}')
print(f'  每折大约包含: {len(X_all) // k_folds} 个样本')
print()

# 存储每折的结果
fold_results = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'confusion_matrix': []
}

# ==================== 第三部分：10折交叉验证训练与测试 ====================
print('=' * 70)
print('开始10折交叉验证')
print('=' * 70)
print()

num_classes = len(np.unique(y_all))
all_classes = np.unique(y_all)

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all), 1):
    print(f'-------- 第 {fold}/{k_folds} 折 --------')
    
    # 划分数据
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
    print(f'  训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本')
    
    # 数据归一化
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    # 训练BP网络
    print('  训练BP网络...')
    bp_net = train_bp_network(X_train_norm, y_train, num_epochs=1000, verbose=False)
    
    # 提取特征
    with torch.no_grad():
        features_train = bp_net(torch.FloatTensor(X_train_norm)).numpy()
        features_test = bp_net(torch.FloatTensor(X_test_norm)).numpy()
    
    # 特征标准化
    feature_scaler = StandardScaler()
    features_train_norm = feature_scaler.fit_transform(features_train)
    features_test_norm = feature_scaler.transform(features_test)
    
    # 训练SVM
    print('  训练SVM分类器...')
    base_svm = SVC(kernel='rbf', C=1.0, gamma='auto', random_state=42)
    svm_model = OneVsOneClassifier(base_svm)
    svm_model.fit(features_train_norm, y_train)
    
    # 测试与评估
    print('  测试模型性能...')
    y_test_pred = svm_model.predict(features_test_norm)
    
    # 计算准确率
    test_accuracy = accuracy_score(y_test, y_test_pred) * 100
    
    # 计算混淆矩阵（确保包含所有类别）
    cm = confusion_matrix(y_test, y_test_pred, labels=all_classes)
    
    # 计算每类的精确率、召回率、F1分数
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        if i < cm.shape[0] and i < cm.shape[1]:
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            
            if (TP + FP) > 0:
                precision[i] = TP / (TP + FP)
            if (TP + FN) > 0:
                recall[i] = TP / (TP + FN)
            if (precision[i] + recall[i]) > 0:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    # 存储结果
    fold_results['accuracy'].append(test_accuracy)
    fold_results['precision'].append(precision)
    fold_results['recall'].append(recall)
    fold_results['f1_score'].append(f1)
    fold_results['confusion_matrix'].append(cm)
    
    print(f'  准确率: {test_accuracy:.2f}%')
    print(f'  平均精确率: {precision.mean():.4f}')
    print(f'  平均召回率: {recall.mean():.4f}')
    print(f'  平均F1分数: {f1.mean():.4f}')
    print()

# ==================== 第四部分：汇总结果 ====================
print('=' * 70)
print('10折交叉验证汇总结果')
print('=' * 70)
print()

# 转换为数组便于计算
accuracy_array = np.array(fold_results['accuracy'])
precision_means = np.array([p.mean() for p in fold_results['precision']])
recall_means = np.array([r.mean() for r in fold_results['recall']])
f1_means = np.array([f.mean() for f in fold_results['f1_score']])

# 计算均值和标准差
mean_accuracy = accuracy_array.mean()
std_accuracy = accuracy_array.std()
mean_precision = precision_means.mean()
std_precision = precision_means.std()
mean_recall = recall_means.mean()
std_recall = recall_means.std()
mean_f1 = f1_means.mean()
std_f1 = f1_means.std()

print('总体平均性能:')
print(f'  准确率: {mean_accuracy:.2f}% (± {std_accuracy:.2f}%)')
print(f'  精确率: {mean_precision:.4f} (± {std_precision:.4f})')
print(f'  召回率: {mean_recall:.4f} (± {std_recall:.4f})')
print(f'  F1分数: {mean_f1:.4f} (± {std_f1:.4f})')
print()

# ==================== 第五部分：详细每折结果 ====================
print('=' * 70)
print('每折详细结果')
print('=' * 70)
print()

print('折数\t准确率(%)\t精确率\t\t召回率\t\tF1分数')
print('-' * 70)
for i in range(k_folds):
    print(f'{i+1}\t{accuracy_array[i]:.2f}\t\t{precision_means[i]:.4f}\t\t'
          f'{recall_means[i]:.4f}\t\t{f1_means[i]:.4f}')
print('-' * 70)
print(f'平均\t{mean_accuracy:.2f}\t\t{mean_precision:.4f}\t\t'
      f'{mean_recall:.4f}\t\t{mean_f1:.4f}')
print(f'标准差\t{std_accuracy:.2f}\t\t{std_precision:.4f}\t\t'
      f'{std_recall:.4f}\t\t{std_f1:.4f}')
print()

# ==================== 第六部分：每类性能统计 ====================
print('=' * 70)
print('各故障类型平均性能 (10折平均)')
print('=' * 70)
print()

# 计算每类的平均性能
class_precision_avg = np.zeros(num_classes)
class_recall_avg = np.zeros(num_classes)
class_f1_avg = np.zeros(num_classes)

for i in range(num_classes):
    precision_sum = sum([fold_results['precision'][fold][i] for fold in range(k_folds)])
    recall_sum = sum([fold_results['recall'][fold][i] for fold in range(k_folds)])
    f1_sum = sum([fold_results['f1_score'][fold][i] for fold in range(k_folds)])
    
    class_precision_avg[i] = precision_sum / k_folds
    class_recall_avg[i] = recall_sum / k_folds
    class_f1_avg[i] = f1_sum / k_folds

print('故障类型\t精确率\t\t召回率\t\tF1分数')
print('-' * 70)
for i in range(num_classes):
    print(f'y{all_classes[i]}\t\t{class_precision_avg[i]:.4f}\t\t'
          f'{class_recall_avg[i]:.4f}\t\t{class_f1_avg[i]:.4f}')
print()

# ==================== 第七部分：可视化结果 ====================
print('生成可视化结果...')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 10))

# 1. 每折准确率折线图
ax1 = plt.subplot(2, 3, 1)
plt.plot(range(1, k_folds+1), accuracy_array, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=mean_accuracy, color='r', linestyle='--', linewidth=2, label='Mean')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Fold')
plt.legend()
plt.grid(True)
plt.ylim([accuracy_array.min()-5, 105])

# 2. 性能指标对比柱状图
ax2 = plt.subplot(2, 3, 2)
metrics = [mean_precision, mean_recall, mean_f1, mean_accuracy/100]
metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
bars = plt.bar(metric_names, metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Score')
plt.title('Average Performance Metrics')
plt.ylim([0, 1.05])
plt.grid(True, axis='y')

for bar, metric in zip(bars, metrics):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{metric:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. 各类性能对比
ax3 = plt.subplot(2, 3, 3)
x = np.arange(num_classes)
width = 0.25
plt.bar(x - width, class_precision_avg, width, label='Precision', color='steelblue')
plt.bar(x, class_recall_avg, width, label='Recall', color='orange')
plt.bar(x + width, class_f1_avg, width, label='F1-Score', color='green')
plt.xlabel('Fault Type')
plt.ylabel('Score')
plt.title('Performance by Fault Type')
plt.xticks(x, [f'y{c}' for c in all_classes])
plt.legend()
plt.ylim([0, 1.05])
plt.grid(True, axis='y')

# 4. 混淆矩阵热图（最后一折）
ax4 = plt.subplot(2, 3, 4)
sns.heatmap(fold_results['confusion_matrix'][-1], annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Fold {k_folds})')

# 5. 箱线图
ax5 = plt.subplot(2, 3, 5)
data_to_plot = [precision_means, recall_means, f1_means, accuracy_array/100]
plt.boxplot(data_to_plot, labels=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
plt.ylabel('Score')
plt.title('Performance Metrics Distribution')
plt.grid(True, axis='y')
plt.ylim([0, 1.05])

# 6. 各折指标变化趋势
ax6 = plt.subplot(2, 3, 6)
plt.plot(range(1, k_folds+1), precision_means, 'r-o', linewidth=2, 
         markersize=6, label='Precision')
plt.plot(range(1, k_folds+1), recall_means, 'g-s', linewidth=2, 
         markersize=6, label='Recall')
plt.plot(range(1, k_folds+1), f1_means, 'b-^', linewidth=2, 
         markersize=6, label='F1-Score')
plt.xlabel('Fold Number')
plt.ylabel('Score')
plt.title('Metrics Trend Across Folds')
plt.legend()
plt.grid(True)
plt.ylim([0, 1.05])

plt.tight_layout()
plt.savefig('BP_SVM_10Fold_Results.png', dpi=300, bbox_inches='tight')
print('  可视化图表已保存: BP_SVM_10Fold_Results.png')
print()

# ==================== 第八部分：保存结果 ====================
print('保存交叉验证结果...')

import joblib
joblib.dump({
    'fold_results': fold_results,
    'k_folds': k_folds,
    'mean_accuracy': mean_accuracy,
    'std_accuracy': std_accuracy,
    'mean_precision': mean_precision,
    'std_precision': std_precision,
    'mean_recall': mean_recall,
    'std_recall': std_recall,
    'mean_f1': mean_f1,
    'std_f1': std_f1,
    'class_precision_avg': class_precision_avg,
    'class_recall_avg': class_recall_avg,
    'class_f1_avg': class_f1_avg
}, 'bp_svm_10fold_results.pkl')

print('  结果已保存: bp_svm_10fold_results.pkl')
print()

print('=' * 70)
print('10折交叉验证完成！')
print('=' * 70)