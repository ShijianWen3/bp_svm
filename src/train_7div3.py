"""
BP-SVM采煤机故障诊断系统 - Python实现
包括：数据加载、7:3随机划分、BP特征提取、SVM分类、结果评估
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report, roc_curve, auc)
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

print('=' * 60)
print('BP-SVM采煤机故障诊断系统 (Python版)')
print('=' * 60)
print()

# ==================== 第一部分：数据加载与合并 ====================
print('步骤1: 加载并合并数据集...')

# 读取训练集和测试集
train_data = pd.read_csv('./datasets/train_data.txt', sep='\t')
test_data = pd.read_csv('./datasets/test_data.txt', sep='\t')

# 合并为总数据集
all_data = pd.concat([train_data, test_data], ignore_index=True)

# 分离特征和标签
X_all = all_data.iloc[:, :6].values  # 前6列是特征
y_all = all_data.iloc[:, 6].values   # 第7列是标签

print(f'  总样本数: {len(X_all)}')
print(f'  特征维度: {X_all.shape[1]}')
print(f'  故障类型: {len(np.unique(y_all))} 类')
print()

# ==================== 第二部分：7:3随机划分 ====================
print('步骤2: 按7:3随机划分训练集和测试集...')

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, 
    test_size=0.3, 
    random_state=42,
    stratify=y_all  # 保持类别比例
)

print(f'  训练集: {len(X_train)} 样本 ({len(X_train)/len(X_all)*100:.1f}%)')
print(f'  测试集: {len(X_test)} 样本 ({len(X_test)/len(X_all)*100:.1f}%)')
print()

# ==================== 第三部分：数据归一化 ====================
print('步骤3: 数据归一化...')

# Min-Max归一化
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

print('  归一化完成')
print()

# ==================== 第四部分：BP神经网络定义 ====================
print('步骤4: 构建BP神经网络...')

class BPNetwork(nn.Module):
    """BP神经网络用于特征提取"""
    def __init__(self, input_size=6, hidden_size=12, output_size=6):
        super(BPNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()  # 对应MATLAB的tansig
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)  # 线性输出
        return x

# 创建网络
input_size = 6
hidden_size = 12
output_size = 6

bp_net = BPNetwork(input_size, hidden_size, output_size)

print(f'  网络结构: {input_size} -> {hidden_size} -> {output_size}')
print(f'  激活函数: Tanh (改进Sigmoid)')
print(f'  训练算法: Adam优化器')
print()

# ==================== 第五部分：训练BP网络提取特征 ====================
print('步骤5: 训练BP网络提取特征...')

# 使用PCA结果作为训练目标
pca = PCA(n_components=output_size)
target_features = pca.fit_transform(X_train_norm)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_norm)
target_tensor = torch.FloatTensor(target_features)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, target_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(bp_net.parameters(), lr=0.01)

# 训练网络
print('  开始训练BP网络...')
num_epochs = 1000
bp_net.train()
train_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = bp_net(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f'    Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')

print('  训练完成！')
print()

# ==================== 第六部分：提取特征 ====================
print('步骤6: 提取训练集和测试集特征...')

bp_net.eval()
with torch.no_grad():
    features_train = bp_net(X_train_tensor).numpy()
    features_test = bp_net(torch.FloatTensor(X_test_norm)).numpy()

print(f'  训练集特征维度: {features_train.shape}')
print(f'  测试集特征维度: {features_test.shape}')
print()

# ==================== 第七部分：特征标准化 ====================
print('步骤7: 特征批标准化...')

feature_scaler = StandardScaler()
features_train_norm = feature_scaler.fit_transform(features_train)
features_test_norm = feature_scaler.transform(features_test)

print('  特征标准化完成')
print()

# ==================== 第八部分：SVM分类器训练 ====================
print('步骤8: 训练SVM分类器...')

# 使用RBF核的SVM
base_svm = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True, random_state=42)

# 一对一多分类策略
svm_model = OneVsOneClassifier(base_svm)

print('  正在训练SVM...')
svm_model.fit(features_train_norm, y_train)

print('  SVM训练完成！')
print()

# ==================== 第九部分：模型测试与评估 ====================
print('步骤9: 模型测试与评估...')
print()

# 训练集预测
y_train_pred = svm_model.predict(features_train_norm)
train_accuracy = accuracy_score(y_train, y_train_pred) * 100

# 测试集预测
y_test_pred = svm_model.predict(features_test_norm)
test_accuracy = accuracy_score(y_test, y_test_pred) * 100

print('=' * 60)
print('模型性能评估结果')
print('=' * 60)
print(f'训练集准确率: {train_accuracy:.2f}%')
print(f'测试集准确率: {test_accuracy:.2f}%')
print('=' * 60)
print()

# ==================== 第十部分：详细分类报告 ====================
print('=' * 60)
print('详细分类报告（测试集）')
print('=' * 60)
print()

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)

# 计算每类的指标
num_classes = len(np.unique(y_all))
precision = precision_score(y_test, y_test_pred, average=None, zero_division=0)
recall = recall_score(y_test, y_test_pred, average=None, zero_division=0)
f1 = f1_score(y_test, y_test_pred, average=None, zero_division=0)

# 打印每类的性能指标
print('故障类型\t样本数\t精确率\t召回率\tF1分数')
print('-' * 60)
for i in range(num_classes):
    num_samples = np.sum(y_test == (i + 1))
    print(f'y{i+1}\t\t{num_samples}\t{precision[i]:.3f}\t{recall[i]:.3f}\t{f1[i]:.3f}')

print(f'\n平均性能:\t\t{precision.mean():.3f}\t{recall.mean():.3f}\t{f1.mean():.3f}')
print()

# ==================== 第十一部分：可视化结果 ====================
print('步骤10: 生成可视化结果...')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig = plt.figure(figsize=(16, 10))

# 1. 训练过程曲线
ax1 = plt.subplot(2, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('BP Network Training Process')
plt.grid(True)

# 2. 混淆矩阵热图
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Test Set)')

# 3. 实际值 vs 预测值对比
ax3 = plt.subplot(2, 3, 3)
plt.plot(range(len(y_test)), y_test, 'bo-', linewidth=1.5, markersize=6, label='True')
plt.plot(range(len(y_test)), y_test_pred, 'rx-', linewidth=1.5, markersize=8, label='Predicted')
plt.xlabel('Test Sample Index')
plt.ylabel('Fault Type')
plt.title('Prediction Comparison (Test Set)')
plt.legend()
plt.grid(True)

# 4. 各类准确率柱状图
ax4 = plt.subplot(2, 3, 4)
class_accuracy = []
for i in range(num_classes):
    class_idx = y_test == (i + 1)
    if np.sum(class_idx) > 0:
        acc = np.sum(y_test_pred[class_idx] == (i + 1)) / np.sum(class_idx) * 100
        class_accuracy.append(acc)
    else:
        class_accuracy.append(0)

plt.bar(range(1, num_classes + 1), class_accuracy, color='steelblue')
plt.xlabel('Fault Type')
plt.ylabel('Accuracy (%)')
plt.title('Classification Accuracy by Fault Type')
plt.xticks(range(1, num_classes + 1), [f'y{i+1}' for i in range(num_classes)])
plt.ylim([0, 105])
plt.grid(True, axis='y')

# 5. 特征空间可视化（PCA降维到2D）
ax5 = plt.subplot(2, 3, 5)
pca_vis = PCA(n_components=2)
features_2d = pca_vis.fit_transform(features_test_norm)
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y_test, cmap='tab10', s=50)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Visualization of Extracted Features')
plt.colorbar(scatter, label='Fault Type')
plt.grid(True)

# 6. 性能指标对比柱状图
ax6 = plt.subplot(2, 3, 6)
metrics = [precision.mean(), recall.mean(), f1.mean(), test_accuracy/100]
metric_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
bars = plt.bar(metric_names, metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylabel('Score')
plt.title('Average Performance Metrics')
plt.ylim([0, 1.05])
plt.grid(True, axis='y')

# 在柱状图上显示数值
for bar, metric in zip(bars, metrics):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{metric:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('BP_SVM_Results.png', dpi=300, bbox_inches='tight')
print('  可视化图表已保存: BP_SVM_Results.png')
print()

# ==================== 第十二部分：错误样本分析 ====================
print('=' * 60)
print('错误样本分析')
print('=' * 60)

error_idx = np.where(y_test_pred != y_test)[0]
num_errors = len(error_idx)

print(f'错误样本数: {num_errors} / {len(y_test)} ({num_errors/len(y_test)*100:.2f}%)')
print()

if num_errors > 0:
    print('样本编号\t真实类型\t预测类型')
    print('-' * 60)
    for i in range(min(10, num_errors)):
        idx = error_idx[i]
        print(f'{idx}\t\ty{y_test[idx]}\t\ty{y_test_pred[idx]}')
    if num_errors > 10:
        print(f'... (还有 {num_errors - 10} 个错误样本未显示)')

print()
print('=' * 60)

# ==================== 第十三部分：保存模型 ====================
print('步骤11: 保存模型...')

# 保存PyTorch模型
torch.save(bp_net.state_dict(), 'bp_network.pth')

# 保存其他模型和参数
import joblib
joblib.dump({
    'svm_model': svm_model,
    'scaler': scaler,
    'feature_scaler': feature_scaler,
    'train_accuracy': train_accuracy,
    'test_accuracy': test_accuracy,
    'num_classes': num_classes
}, 'bp_svm_model.pkl')

print('  模型已保存:')
print('    - bp_network.pth (BP网络权重)')
print('    - bp_svm_model.pkl (SVM模型和标准化器)')
print()

# ==================== 第十四部分：模型使用示例 ====================
print('=' * 60)
print('模型使用示例')
print('=' * 60)
print()

print('# 加载模型')
print('import torch')
print('import joblib')
print()
print('# 加载BP网络')
print("bp_net = BPNetwork(6, 12, 6)")
print("bp_net.load_state_dict(torch.load('bp_network.pth'))")
print("bp_net.eval()")
print()
print('# 加载SVM和标准化器')
print("models = joblib.load('bp_svm_model.pkl')")
print()
print('# 对新样本进行预测')
print("new_sample = np.array([[0.85, 0.12, 0.08, 0.23, 0.15, 0.09]])")
print()
print('# 1. 归一化')
print("new_sample_norm = models['scaler'].transform(new_sample)")
print()
print('# 2. BP提取特征')
print("with torch.no_grad():")
print("    new_features = bp_net(torch.FloatTensor(new_sample_norm)).numpy()")
print()
print('# 3. 特征标准化')
print("new_features_norm = models['feature_scaler'].transform(new_features)")
print()
print('# 4. SVM预测')
print("prediction = models['svm_model'].predict(new_features_norm)")
print("print(f'预测故障类型: y{prediction[0]}')")
print()

# 实际演示
print('=' * 60)
print('实际预测演示（使用测试集第1个样本）')
print('=' * 60)

demo_sample = X_test[0:1]
demo_sample_norm = scaler.transform(demo_sample)
with torch.no_grad():
    demo_features = bp_net(torch.FloatTensor(demo_sample_norm)).numpy()
demo_features_norm = feature_scaler.transform(demo_features)
demo_prediction = svm_model.predict(demo_features_norm)[0]

print(f'输入特征: {demo_sample[0]}')
print(f'真实类型: y{y_test[0]}')
print(f'预测类型: y{demo_prediction}')
if demo_prediction == y_test[0]:
    print('预测结果: ✓ 正确')
else:
    print('预测结果: ✗ 错误')

print()
print('=' * 60)
print('BP-SVM故障诊断系统运行完成！')
print('=' * 60)