%% BP-SVM采煤机故障诊断训练与测试代码（优化版）
% 实现论文中的BP-SVM方法
% 包括：数据加载、BP特征提取、SVM分类、结果评估

clear; clc; close all;

%% ==================== 第一部分：数据加载 ====================
fprintf('========================================\n');
fprintf('BP-SVM采煤机故障诊断系统（优化版）\n');
fprintf('========================================\n\n');

fprintf('步骤1: 加载数据...\n');

% 使用 readtable 读取训练数据文件
train_data_table = readtable('../datasets/train_data.txt', 'Delimiter', '\t');
test_data_table = readtable('../datasets/test_data.txt', 'Delimiter', '\t');

% 将读取的 table 格式转换为数值矩阵
train_data_matrix = table2array(train_data_table);
test_data_matrix = table2array(test_data_table);

% 前6列是特征，第7列是标签
X_train = train_data_matrix(:, 1:6);
Y_train = train_data_matrix(:, 7);
X_test = test_data_matrix(:, 1:6);
Y_test = test_data_matrix(:, 7);

fprintf('  训练集: %d 组样本\n', size(X_train, 1));
fprintf('  测试集: %d 组样本\n', size(X_test, 1));
fprintf('  特征维度: %d\n', size(X_train, 2));
fprintf('  故障类型: %d 类\n\n', length(unique(Y_train)));

%% ==================== 第二部分：数据增强（关键优化！） ====================
fprintf('步骤2: 数据增强（SMOTE过采样）...\n');

% 统计每个类别的样本数
unique_labels = unique(Y_train);
class_counts = arrayfun(@(x) sum(Y_train == x), unique_labels);
max_count = max(class_counts);

fprintf('  原始类别分布:\n');
for i = 1:length(unique_labels)
    fprintf('    y%d: %d 组\n', unique_labels(i), class_counts(i));
end

% 对少数类进行过采样（添加高斯噪声）
X_train_aug = X_train;
Y_train_aug = Y_train;

for i = 1:length(unique_labels)
    label = unique_labels(i);
    label_idx = find(Y_train == label);
    current_count = length(label_idx);
    
    % 如果样本数少于最大值的50%，进行过采样
    if current_count < max_count * 0.5
        need_samples = round(max_count * 0.5) - current_count;
        
        % 通过添加噪声生成新样本
        for j = 1:need_samples
            % 随机选择一个同类样本
            src_idx = label_idx(randi(length(label_idx)));
            new_sample = X_train(src_idx, :) + 0.05 * randn(1, 6);  % 添加5%噪声
            new_sample = max(0, min(1, new_sample));  % 裁剪到[0,1]
            
            X_train_aug = [X_train_aug; new_sample];
            Y_train_aug = [Y_train_aug; label];
        end
    end
end

% 打乱增强后的数据
shuffle_idx = randperm(length(Y_train_aug));
X_train = X_train_aug(shuffle_idx, :);
Y_train = Y_train_aug(shuffle_idx);

fprintf('  增强后训练集: %d 组样本\n', size(X_train, 1));
fprintf('  增强后类别分布:\n');
for i = 1:length(unique_labels)
    fprintf('    y%d: %d 组\n', unique_labels(i), sum(Y_train == unique_labels(i)));
end
fprintf('\n');

%% ==================== 第三部分：数据归一化 ====================
fprintf('步骤3: 数据归一化...\n');

% Min-Max归一化
X_min = min(X_train, [], 1);
X_max = max(X_train, [], 1);
X_range = X_max - X_min;
X_range(X_range == 0) = 1;  % 避免除零

X_train_norm = (X_train - X_min) ./ X_range;
X_test_norm = (X_test - X_min) ./ X_range;

fprintf('  归一化完成\n\n');

%% ==================== 第四部分：BP神经网络特征提取 ====================
fprintf('步骤4: 构建BP神经网络...\n');

% BP网络参数设置（优化后）
input_size = 6;      % 输入层：6个特征
hidden_size = 20;    % 隐藏层：20个神经元（增加了）
output_size = 6;     % 最终提取的特征维度

% 检查并重新映射标签（确保从1开始连续）
unique_labels = unique(Y_train);
num_classes = length(unique_labels);

% 创建标签映射
label_map = containers.Map(unique_labels, 1:num_classes);
Y_train_mapped = zeros(size(Y_train));
Y_test_mapped = zeros(size(Y_test));

% 映射训练集标签
for i = 1:length(Y_train)
    Y_train_mapped(i) = label_map(Y_train(i));
end
% 映射测试集标签
for i = 1:length(Y_test)
    Y_test_mapped(i) = label_map(Y_test(i));
end

fprintf('  标签映射: ');
for i = 1:num_classes
    fprintf('%d->%d ', unique_labels(i), i);
end
fprintf('\n');

% 创建BP神经网络
net = feedforwardnet(hidden_size, 'trainbr');  % 使用贝叶斯正则化（更好的泛化）

% 网络配置（优化后）
net.trainParam.epochs = 500;         % 减少训练轮数防止过拟合
net.trainParam.goal = 1e-4;          % 放宽目标误差
net.trainParam.showWindow = false;   % 关闭训练窗口加快速度
net.trainParam.show = 100;           % 减少显示频率

% 使用验证集进行早停
net.divideParam.trainRatio = 0.8;    % 80%训练
net.divideParam.valRatio = 0.2;      % 20%验证（早停）
net.divideParam.testRatio = 0;

% 设置激活函数
net.layers{1}.transferFcn = 'tansig';   % 隐藏层
net.layers{2}.transferFcn = 'softmax';  % 输出层
net.layers{2}.size = num_classes;

fprintf('  网络结构: %d -> %d -> %d\n', input_size, hidden_size, num_classes);
fprintf('  激活函数: 隐藏层=tansig, 输出层=softmax\n');
fprintf('  训练算法: Bayesian Regularization (trainbr)\n\n');

%% ==================== 第五部分：训练BP网络（有监督分类） ====================
fprintf('步骤5: 训练BP网络（有监督分类）...\n');

% 将标签转换为one-hot编码
target_features = full(ind2vec(Y_train_mapped'));

% 转置数据以符合Matlab格式（列为样本）
X_train_net = X_train_norm';

% 训练BP网络
fprintf('  开始训练BP网络（分类任务）...\n');
tic;
[net, tr] = train(net, X_train_net, target_features);
training_time = toc;
fprintf('  训练完成！用时: %.2f 秒\n\n', training_time);

%% ==================== 第六部分：提取隐藏层特征 ====================
fprintf('步骤6: 提取隐藏层特征（而非输出层）...\n');

% 手动前向传播到隐藏层
hidden_weights = net.IW{1,1};  % 输入层→隐藏层权重
hidden_bias = net.b{1};         % 隐藏层偏置

% 提取训练集隐藏层特征
hidden_input_train = hidden_weights * X_train_norm' + hidden_bias;
features_train = tansig(hidden_input_train)';

% 提取测试集隐藏层特征
hidden_input_test = hidden_weights * X_test_norm' + hidden_bias;
features_test = tansig(hidden_input_test)';

fprintf('  训练集特征维度: %d x %d\n', size(features_train, 1), size(features_train, 2));
fprintf('  测试集特征维度: %d x %d\n', size(features_test, 1), size(features_test, 2));
fprintf('  特征来源: BP隐藏层激活值\n\n');

%% ==================== 第七部分：特征标准化（批归一化） ====================
fprintf('步骤7: 特征批标准化...\n');

% 对提取的特征进行标准化（均值0，方差1）
features_mean = mean(features_train, 1);
features_std = std(features_train, 0, 1);
features_std(features_std == 0) = 1;  % 避免除零

features_train_norm = (features_train - features_mean) ./ features_std;
features_test_norm = (features_test - features_mean) ./ features_std;

fprintf('  特征标准化完成\n\n');

%% ==================== 第八部分：SVM分类器训练（多核融合） ====================
fprintf('步骤8: 训练SVM分类器（集成多核）...\n');

% 尝试多种核函数，选择最优
kernels = {'linear', 'rbf', 'polynomial'};
cv_accuracies = zeros(1, length(kernels));
models = cell(1, length(kernels));

fprintf('  测试不同核函数:\n');
for k = 1:length(kernels)
    template = templateSVM(...
        'KernelFunction', kernels{k}, ...
        'BoxConstraint', 10, ...
        'Standardize', false);
    
    % 5折交叉验证
    cv_model = fitcecoc(features_train_norm, Y_train_mapped, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'KFold', 5);
    
    cv_accuracies(k) = 1 - kfoldLoss(cv_model);
    fprintf('    %s 核: CV准确率 = %.2f%%\n', kernels{k}, cv_accuracies(k)*100);
    
    % 训练完整模型
    models{k} = fitcecoc(features_train_norm, Y_train_mapped, ...
        'Learners', template, ...
        'Coding', 'onevsone');
end

% 选择最优核函数
[best_cv_acc, best_idx] = max(cv_accuracies);
SVMModel = models{best_idx};

fprintf('\n  最优核函数: %s (CV准确率: %.2f%%)\n', kernels{best_idx}, best_cv_acc*100);

% 使用超参数优化进一步提升（仅对最优核）
fprintf('  对最优核进行超参数优化...\n');
template_opt = templateSVM('KernelFunction', kernels{best_idx}, 'Standardize', false);

tic;
SVMModel = fitcecoc(features_train_norm, Y_train_mapped, ...
    'Learners', template_opt, ...
    'Coding', 'onevsone', ...
    'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
    'HyperparameterOptimizationOptions', struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 20, ...  % 减少到20次加快速度
        'ShowPlots', false, ...
        'Verbose', 0));
svm_training_time = toc;

fprintf('  SVM训练完成！用时: %.2f 秒\n\n', svm_training_time);


%% 在第九部分之前加入：集成多个模型
fprintf('步骤8.5: 集成学习（Bagging）...\n');

n_models = 5;
all_predictions = zeros(length(Y_test_mapped), n_models);

for m = 1:n_models
    fprintf('  训练第 %d/%d 个模型...\n', m, n_models);
    
    % Bootstrap采样
    n_samples = length(Y_train_mapped);
    bootstrap_idx = randi(n_samples, n_samples, 1);
    
    % 用采样数据训练
    X_boot = features_train_norm(bootstrap_idx, :);
    Y_boot = Y_train_mapped(bootstrap_idx);
    
    svm_temp = fitcecoc(X_boot, Y_boot, ...
        'Learners', templateSVM('KernelFunction', 'rbf'), ...
        'Coding', 'onevsone');
    
    all_predictions(:, m) = predict(svm_temp, features_test_norm);
end

% 投票
Y_test_pred = mode(all_predictions, 2);
test_accuracy = sum(Y_test_pred == Y_test_mapped) / length(Y_test_mapped) * 100;
fprintf('  集成后测试准确率: %.2f%%\n\n', test_accuracy);


%% ==================== 第九部分：模型测试与评估 ====================
fprintf('步骤9: 模型测试与评估...\n\n');

% 训练集预测
Y_train_pred = predict(SVMModel, features_train_norm);
train_accuracy = sum(Y_train_pred == Y_train_mapped) / length(Y_train_mapped) * 100;

% 测试集预测
Y_test_pred = predict(SVMModel, features_test_norm);
test_accuracy = sum(Y_test_pred == Y_test_mapped) / length(Y_test_mapped) * 100;

fprintf('========================================\n');
fprintf('模型性能评估结果\n');
fprintf('========================================\n');
fprintf('训练集准确率: %.2f%%\n', train_accuracy);
fprintf('测试集准确率: %.2f%%\n', test_accuracy);
fprintf('训练总用时: %.2f 秒\n', training_time + svm_training_time);
fprintf('========================================\n\n');

% 显示混淆矩阵
fprintf('测试集混淆矩阵:\n');
C = confusionmat(Y_test_mapped, Y_test_pred);
fprintf('     ');
for i = 1:num_classes
    fprintf('y%d   ', i);
end
fprintf('\n');
for i = 1:num_classes
    fprintf('y%d: ', i);
    for j = 1:num_classes
        fprintf('%3d  ', C(i,j));
    end
    fprintf('\n');
end

% 计算每个类别的准确率
fprintf('\n各类别准确率:\n');
for i = 1:num_classes
    class_acc = C(i,i) / sum(C(i,:)) * 100;
    fprintf('  y%d: %.2f%%\n', i, class_acc);
end