"""
BP-SVM采煤机故障诊断 - 数据生成脚本 (Python版本)
根据论文表1和表2生成模拟故障数据集，并保存为txt文件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 设置随机种子，保证可重复性
np.random.seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_fault_data():
    """
    生成采煤机故障数据集
    """
    
    # 1. 定义故障规则（根据论文表1）
    # 故障现象 -> 可能的故障原因映射
    fault_rules = {
        'x1': [1, 2, 3, 6],  # 泵压力异常 -> y1,y2,y3,y6
        'x2': [3, 6],        # 泵冷却压力异常 -> y3,y6
        'x3': [1, 6],        # 泵工作转速异常 -> y1,y6
        'x4': [4, 6],        # 电动机温度异常 -> y4,y6
        'x5': [5],           # 电动机电流异常 -> y5
        'x6': [4, 5, 6, 7, 8]  # 电动机转速异常 -> y4,y5,y6,y7,y8
    }
    
    # 2. 定义各故障现象的样本数量（根据论文表2）
    sample_counts = {
        'x1': 60,
        'x2': 26,
        'x3': 25,
        'x4': 33,
        'x5': 16,
        'x6': 45
    }
    
    total_samples = sum(sample_counts.values())
    
    # 3. 初始化数据
    X_data = []
    Y_data = []
    
    # 4. 为每种故障现象生成数据
    for fault_feature, num_samples in sample_counts.items():
        feature_idx = int(fault_feature[1]) - 1  # x1->0, x2->1, ...
        
        for _ in range(num_samples):
            # 生成6维特征向量
            feature_vector = np.zeros(6)
            
            for j in range(6):
                if j == feature_idx:
                    # 异常特征：高值(0.7-1.0) + 噪声
                    feature_vector[j] = 0.7 + 0.3 * np.random.rand() + 0.05 * np.random.randn()
                else:
                    # 正常特征：低值(0.0-0.3) + 噪声
                    feature_vector[j] = 0.15 * np.random.rand() + 0.02 * np.random.randn()
            
            # 添加故障间的耦合效应
            if fault_feature == 'x1':  # 泵压力异常可能影响转速
                feature_vector[2] += 0.2 * np.random.rand()
            elif fault_feature == 'x4':  # 电动机温度异常可能影响转速
                feature_vector[5] += 0.15 * np.random.rand()
            elif fault_feature == 'x6':  # 转速异常可能影响温度
                feature_vector[3] += 0.1 * np.random.rand()
            
            # 裁剪到[0, 1]范围
            feature_vector = np.clip(feature_vector, 0, 1)
            
            # 根据规则随机选择一个故障原因
            possible_faults = fault_rules[fault_feature]
            fault_label = np.random.choice(possible_faults)
            
            X_data.append(feature_vector)
            Y_data.append(fault_label)
    
    # 5. 转换为numpy数组
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    
    # 6. 打乱数据顺序
    shuffle_idx = np.random.permutation(total_samples)
    X_data = X_data[shuffle_idx]
    Y_data = Y_data[shuffle_idx]
    
    return X_data, Y_data, sample_counts


def split_and_normalize(X_data, Y_data, train_size=140):
    """
    划分训练集和测试集，并进行归一化
    """
    # 划分数据集
    X_train = X_data[:train_size]
    Y_train = Y_data[:train_size]
    X_test = X_data[train_size:]
    Y_test = Y_data[train_size:]
    
    # Min-Max归一化
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # 避免除零
    
    X_train_norm = (X_train - X_min) / X_range
    X_test_norm = (X_test - X_min) / X_range
    
    return X_train, Y_train, X_test, Y_test, X_train_norm, X_test_norm, X_min, X_max


def save_data_to_txt(X_train, Y_train, X_test, Y_test, 
                     X_train_norm, X_test_norm, sample_counts):
    """
    保存数据为txt文件（多种格式）
    """
    
    # 1. 保存原始训练数据
    train_data = np.column_stack((X_train, Y_train))
    np.savetxt('train_data.txt', train_data, fmt='%.6f', delimiter='\t',
               header='x1\tx2\tx3\tx4\tx5\tx6\tfault_label', comments='')
    
    # 2. 保存原始测试数据
    test_data = np.column_stack((X_test, Y_test))
    np.savetxt('test_data.txt', test_data, fmt='%.6f', delimiter='\t',
               header='x1\tx2\tx3\tx4\tx5\tx6\tfault_label', comments='')
    
    # 3. 保存归一化训练数据
    train_data_norm = np.column_stack((X_train_norm, Y_train))
    np.savetxt('train_data_normalized.txt', train_data_norm, fmt='%.6f', delimiter='\t',
               header='x1_norm\tx2_norm\tx3_norm\tx4_norm\tx5_norm\tx6_norm\tfault_label', comments='')
    
    # 4. 保存归一化测试数据
    test_data_norm = np.column_stack((X_test_norm, Y_test))
    np.savetxt('test_data_normalized.txt', test_data_norm, fmt='%.6f', delimiter='\t',
               header='x1_norm\tx2_norm\tx3_norm\tx4_norm\tx5_norm\tx6_norm\tfault_label', comments='')
    
    # 5. 保存数据说明文档
    with open('data_description.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("采煤机故障诊断数据集说明\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 数据集概况\n")
        f.write("-" * 60 + "\n")
        f.write(f"总样本数: {len(X_train) + len(X_test)}\n")
        f.write(f"训练集: {len(X_train)} 组 ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)\n")
        f.write(f"测试集: {len(X_test)} 组 ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)\n")
        f.write(f"特征维度: 6\n")
        f.write(f"故障类型: 8 类\n\n")
        
        f.write("2. 特征说明（输入特征）\n")
        f.write("-" * 60 + "\n")
        f.write("x1: 泵压力异常\n")
        f.write("x2: 泵冷却压力异常\n")
        f.write("x3: 泵工作转速异常\n")
        f.write("x4: 电动机温度异常\n")
        f.write("x5: 电动机电流异常\n")
        f.write("x6: 电动机转速异常\n\n")
        
        f.write("3. 故障类型说明（输出标签）\n")
        f.write("-" * 60 + "\n")
        f.write("y1 (1): 液压泵故障 - 泵体磨损、压力不足、流量异常\n")
        f.write("y2 (2): 液压缸故障 - 密封件损坏、活塞杆弯曲、缸体泄漏\n")
        f.write("y3 (3): 液压阀故障 - 阀芯卡滞、阀体堵塞\n")
        f.write("y4 (4): 电动机故障 - 过热、绝缘损坏、绕组短路\n")
        f.write("y5 (5): 电缆故障 - 断裂、绝缘层损坏、接头松动\n")
        f.write("y6 (6): 控制系统故障 - PLC模块故障、传感器失灵、控制线路接触不良\n")
        f.write("y7 (7): 截割部故障 - 截齿磨损、截割电动机损坏、截割滚筒轴承失效\n")
        f.write("y8 (8): 牵引部故障 - 牵引链条断裂、牵引电动机故障、行走轮磨损\n\n")
        
        f.write("4. 故障规则（论文表1）\n")
        f.write("-" * 60 + "\n")
        f.write("x1异常 → y1, y2, y3, y6\n")
        f.write("x2异常 → y3, y6\n")
        f.write("x3异常 → y1, y6\n")
        f.write("x4异常 → y4, y6\n")
        f.write("x5异常 → y5\n")
        f.write("x6异常 → y4, y5, y6, y7, y8\n\n")
        
        f.write("5. 故障现象样本分布（论文表2）\n")
        f.write("-" * 60 + "\n")
        for feature, count in sample_counts.items():
            f.write(f"{feature}: {count} 组\n")
        f.write("\n")
        
        f.write("6. 故障类型样本分布\n")
        f.write("-" * 60 + "\n")
        all_Y = np.concatenate([Y_train, Y_test])
        for i in range(1, 9):
            count = np.sum(all_Y == i)
            percentage = count / len(all_Y) * 100
            f.write(f"y{i}: {count} 组 ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("7. 文件说明\n")
        f.write("-" * 60 + "\n")
        f.write("train_data.txt - 原始训练数据\n")
        f.write("test_data.txt - 原始测试数据\n")
        f.write("train_data_normalized.txt - 归一化训练数据\n")
        f.write("test_data_normalized.txt - 归一化测试数据\n")
        f.write("data_description.txt - 本说明文件\n\n")
        
        f.write("8. 数据格式\n")
        f.write("-" * 60 + "\n")
        f.write("每行代表一个样本，列之间用制表符分隔\n")
        f.write("前6列为特征值（x1-x6），最后1列为故障标签（1-8）\n")
        f.write("归一化数据使用Min-Max方法，范围为[0, 1]\n\n")
        
        f.write("=" * 60 + "\n")


def visualize_data(X_train, Y_train, X_test, Y_test, X_train_norm, sample_counts):
    """
    数据可视化
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 故障类型分布
    plt.subplot(2, 3, 1)
    all_Y = np.concatenate([Y_train, Y_test])
    plt.hist(all_Y, bins=np.arange(0.5, 9.5, 1), color='steelblue', edgecolor='black')
    plt.xlabel('Fault Type (y1-y8)')
    plt.ylabel('Sample Count')
    plt.title('Fault Type Distribution')
    plt.xticks(range(1, 9), [f'y{i}' for i in range(1, 9)])
    plt.grid(True, alpha=0.3)
    
    # 2. 故障现象分布
    plt.subplot(2, 3, 2)
    features = list(sample_counts.keys())
    counts = list(sample_counts.values())
    plt.bar(features, counts, color='coral', edgecolor='black')
    plt.xlabel('Fault Symptom')
    plt.ylabel('Sample Count')
    plt.title('Fault Symptom Distribution (Table 2)')
    plt.grid(True, alpha=0.3)
    
    # 3. 特征相关性热图
    plt.subplot(2, 3, 3)
    all_X = np.concatenate([X_train, X_test])
    corr_matrix = np.corrcoef(all_X.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                xticklabels=[f'x{i}' for i in range(1, 7)],
                yticklabels=[f'x{i}' for i in range(1, 7)])
    plt.title('Feature Correlation Matrix')
    
    # 4. 训练集特征箱线图
    plt.subplot(2, 3, 4)
    plt.boxplot(X_train_norm, labels=[f'x{i}' for i in range(1, 7)])
    plt.ylabel('Normalized Feature Value')
    plt.title('Training Set Feature Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. PCA降维可视化
    plt.subplot(2, 3, 5)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_norm)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_train, cmap='tab10', 
                         edgecolors='black', alpha=0.7)
    plt.colorbar(scatter, ticks=range(1, 9))
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('PCA Visualization (Training Set)')
    plt.grid(True, alpha=0.3)
    
    # 6. 数据集统计信息
    plt.subplot(2, 3, 6)
    plt.axis('off')
    stats_text = f"""
    Dataset Statistics
    ─────────────────────────
    Total Samples: {len(all_Y)}
    Training Set: {len(Y_train)} ({len(Y_train)/len(all_Y)*100:.1f}%)
    Test Set: {len(Y_test)} ({len(Y_test)/len(all_Y)*100:.1f}%)
    
    Fault Type Distribution:
    """
    for i in range(1, 9):
        count = np.sum(all_Y == i)
        stats_text += f"\n    y{i}: {count} ({count/len(all_Y)*100:.1f}%)"
    
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
             family='monospace')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存: data_visualization.png")
    plt.show()


def main():
    """
    主函数
    """
    print("=" * 60)
    print("BP-SVM采煤机故障诊断数据生成器")
    print("=" * 60)
    print("\n正在生成数据...")
    
    # 1. 生成数据
    X_data, Y_data, sample_counts = generate_fault_data()
    print(f"✓ 已生成 {len(X_data)} 组样本")
    
    # 2. 划分和归一化
    X_train, Y_train, X_test, Y_test, X_train_norm, X_test_norm, X_min, X_max = \
        split_and_normalize(X_data, Y_data)
    print(f"✓ 训练集: {len(X_train)} 组")
    print(f"✓ 测试集: {len(X_test)} 组")
    
    # 3. 保存数据
    print("\n正在保存数据文件...")
    save_data_to_txt(X_train, Y_train, X_test, Y_test, 
                     X_train_norm, X_test_norm, sample_counts)
    print("✓ train_data.txt")
    print("✓ test_data.txt")
    print("✓ train_data_normalized.txt")
    print("✓ test_data_normalized.txt")
    print("✓ data_description.txt")
    
    # 4. 数据可视化
    print("\n正在生成可视化图表...")
    visualize_data(X_train, Y_train, X_test, Y_test, X_train_norm, sample_counts)
    
    # 5. 打印统计信息
    print("\n" + "=" * 60)
    print("数据统计信息")
    print("=" * 60)
    print(f"\n总样本数: {len(X_data)}")
    print(f"训练集: {len(X_train)} 组 ({len(X_train)/len(X_data)*100:.1f}%)")
    print(f"测试集: {len(X_test)} 组 ({len(X_test)/len(X_data)*100:.1f}%)")
    print(f"\n故障类型分布:")
    for i in range(1, 9):
        count = np.sum(Y_data == i)
        print(f"  y{i}: {count:3d} 组 ({count/len(Y_data)*100:5.1f}%)")
    
    # 6. 显示前10组训练样本
    print("\n" + "=" * 60)
    print("前10组训练样本示例")
    print("=" * 60)
    print(f"{'编号':<5} {'x1':<8} {'x2':<8} {'x3':<8} {'x4':<8} {'x5':<8} {'x6':<8} {'故障':<5}")
    print("-" * 60)
    for i in range(min(10, len(X_train))):
        print(f"{i+1:<5}", end=" ")
        for j in range(6):
            print(f"{X_train[i, j]:<8.3f}", end=" ")
        print(f"y{int(Y_train[i]):<5}")
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()