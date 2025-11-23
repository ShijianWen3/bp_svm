%% BP-SVM采煤机故障诊断 - 10折交叉验证
% 实现论文中的BP-SVM方法
% 使用10折交叉验证评估模型性能

clear; clc; close all;

%% ==================== 第一部分：数据加载 ====================
fprintf('========================================\n');
fprintf('BP-SVM采煤机故障诊断系统 (10折交叉验证)\n');
fprintf('========================================\n\n');

fprintf('步骤1: 加载总数据集...\n');

% 读取总数据集 (假设文件名为 all_data.txt 或合并训练集和测试集)
% 如果有总数据集文件，使用下面这行：
% all_data_table = readtable('../datasets/all_data.txt', 'Delimiter', '\t');

% 如果需要合并训练集和测试集：
train_data_table = readtable('../datasets/train_data.txt', 'Delimiter', '\t');
test_data_table = readtable('../datasets/test_data.txt', 'Delimiter', '\t');
all_data_table = [train_data_table; test_data_table];

% 转换为矩阵
all_data_matrix = table2array(all_data_table);

% 分离特征和标签
X_all = all_data_matrix(:, 1:6);  % 前6列是特征
Y_all = all_data_matrix(:, 7);    % 第7列是标签

fprintf('  总样本数: %d\n', size(X_all, 1));
fprintf('  特征维度: %d\n', size(X_all, 2));
fprintf('  故障类型: %d 类\n\n', length(unique(Y_all)));

%% ==================== 第二部分：10折交叉验证设置 ====================
fprintf('步骤2: 设置10折交叉验证...\n');

k_folds = 10;
cv_indices = crossvalind('Kfold', Y_all, k_folds);

fprintf('  交叉验证折数: %d\n', k_folds);
fprintf('  每折大约包含: %d 个样本\n\n', floor(length(Y_all) / k_folds));

% 存储每折的结果
fold_results = struct();
fold_results.accuracy = zeros(k_folds, 1);
fold_results.precision = cell(k_folds, 1);
fold_results.recall = cell(k_folds, 1);
fold_results.f1_score = cell(k_folds, 1);
fold_results.confusion_matrix = cell(k_folds, 1);

%% ==================== 第三部分：10折交叉验证训练与测试 ====================
fprintf('========================================\n');
fprintf('开始10折交叉验证\n');
fprintf('========================================\n\n');

for fold = 1:k_folds
    fprintf('-------- 第 %d/%d 折 --------\n', fold, k_folds);
    
    % 划分训练集和测试集
    test_idx = (cv_indices == fold);
    train_idx = ~test_idx;
    
    X_train = X_all(train_idx, :);
    Y_train = Y_all(train_idx);
    X_test = X_all(test_idx, :);
    Y_test = Y_all(test_idx);
    
    fprintf('  训练集: %d 样本, 测试集: %d 样本\n', ...
            size(X_train, 1), size(X_test, 1));
    
    %% 数据归一化
    X_min = min(X_train, [], 1);
    X_max = max(X_train, [], 1);
    X_range = X_max - X_min;
    X_range(X_range == 0) = 1;
    
    X_train_norm = (X_train - X_min) ./ X_range;
    X_test_norm = (X_test - X_min) ./ X_range;
    
    %% BP神经网络特征提取
    fprintf('  构建BP神经网络...\n');
    
    input_size = 6;
    hidden_size = 12;
    output_size = 6;
    
    net = feedforwardnet(hidden_size, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-5;
    net.trainParam.lr = 0.01;
    net.trainParam.show = 50;
    net.trainParam.showWindow = false;  % 关闭训练窗口以加速
    
    net.divideParam.trainRatio = 1;
    net.divideParam.valRatio = 0;
    net.divideParam.testRatio = 0;
    
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';
    
    %% 训练BP网络
    fprintf('  训练BP网络...\n');
    
    % 使用PCA作为训练目标
    [~, score] = pca(X_train_norm);
    target_features = score(:, 1:output_size)';
    X_train_net = X_train_norm';
    
    [net, ~] = train(net, X_train_net, target_features);
    
    %% 提取特征
    features_train = net(X_train_norm')';
    features_test = net(X_test_norm')';
    
    %% 特征标准化
    features_mean = mean(features_train, 1);
    features_std = std(features_train, 0, 1);
    features_std(features_std == 0) = 1;
    
    features_train_norm = (features_train - features_mean) ./ features_std;
    features_test_norm = (features_test - features_mean) ./ features_std;
    
    %% 训练SVM分类器
    fprintf('  训练SVM分类器...\n');
    
    template = templateSVM(...
        'KernelFunction', 'rbf', ...
        'KernelScale', 'auto', ...
        'BoxConstraint', 1, ...
        'Standardize', false);
    
    SVMModel = fitcecoc(features_train_norm, Y_train, ...
        'Learners', template, ...
        'Coding', 'onevsone', ...
        'Prior', 'uniform');
    
    %% 测试与评估
    fprintf('  测试模型性能...\n');
    
    Y_test_pred = predict(SVMModel, features_test_norm);
    test_accuracy = sum(Y_test_pred == Y_test) / length(Y_test) * 100;
    
    % 获取总类别数（从全体数据）
    num_classes = length(unique(Y_all));
    all_classes = unique(Y_all);
    
    % 计算混淆矩阵，指定所有可能的类别
    C = confusionmat(Y_test, Y_test_pred, 'Order', all_classes);
    
    % 计算每类的精确率、召回率、F1分数
    precision = zeros(num_classes, 1);
    recall = zeros(num_classes, 1);
    f1_score = zeros(num_classes, 1);
    
    for i = 1:num_classes
        % 确保索引不越界
        if i <= size(C, 1) && i <= size(C, 2)
            TP = C(i, i);
            FP = sum(C(:, i)) - TP;
            FN = sum(C(i, :)) - TP;
        else
            % 如果该类别不在当前折中
            TP = 0;
            FP = 0;
            FN = 0;
        end
        
        if (TP + FP) > 0
            precision(i) = TP / (TP + FP);
        else
            precision(i) = 0;
        end
        
        if (TP + FN) > 0
            recall(i) = TP / (TP + FN);
        else
            recall(i) = 0;
        end
        
        if (precision(i) + recall(i)) > 0
            f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
        else
            f1_score(i) = 0;
        end
    end
    
    % 存储结果
    fold_results.accuracy(fold) = test_accuracy;
    fold_results.precision{fold} = precision;
    fold_results.recall{fold} = recall;
    fold_results.f1_score{fold} = f1_score;
    fold_results.confusion_matrix{fold} = C;
    
    fprintf('  准确率: %.2f%%\n', test_accuracy);
    fprintf('  平均精确率: %.4f\n', mean(precision));
    fprintf('  平均召回率: %.4f\n', mean(recall));
    fprintf('  平均F1分数: %.4f\n\n', mean(f1_score));
end

%% ==================== 第四部分：汇总结果 ====================
fprintf('========================================\n');
fprintf('10折交叉验证汇总结果\n');
fprintf('========================================\n\n');

% 计算平均指标
mean_accuracy = mean(fold_results.accuracy);
std_accuracy = std(fold_results.accuracy);

% 汇总每折的平均精确率、召回率、F1分数
all_precision_means = zeros(k_folds, 1);
all_recall_means = zeros(k_folds, 1);
all_f1_means = zeros(k_folds, 1);

for fold = 1:k_folds
    all_precision_means(fold) = mean(fold_results.precision{fold});
    all_recall_means(fold) = mean(fold_results.recall{fold});
    all_f1_means(fold) = mean(fold_results.f1_score{fold});
end

mean_precision = mean(all_precision_means);
std_precision = std(all_precision_means);
mean_recall = mean(all_recall_means);
std_recall = std(all_recall_means);
mean_f1 = mean(all_f1_means);
std_f1 = std(all_f1_means);

fprintf('总体平均性能:\n');
fprintf('  准确率: %.2f%% (± %.2f%%)\n', mean_accuracy, std_accuracy);
fprintf('  精确率: %.4f (± %.4f)\n', mean_precision, std_precision);
fprintf('  召回率: %.4f (± %.4f)\n', mean_recall, std_recall);
fprintf('  F1分数: %.4f (± %.4f)\n\n', mean_f1, std_f1);

%% ==================== 第五部分：详细每折结果 ====================
fprintf('========================================\n');
fprintf('每折详细结果\n');
fprintf('========================================\n\n');

fprintf('折数\t准确率(%%)\t精确率\t\t召回率\t\tF1分数\n');
fprintf('----------------------------------------------------------------\n');
for fold = 1:k_folds
    fprintf('%d\t%.2f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
            fold, ...
            fold_results.accuracy(fold), ...
            all_precision_means(fold), ...
            all_recall_means(fold), ...
            all_f1_means(fold));
end
fprintf('----------------------------------------------------------------\n');
fprintf('平均\t%.2f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        mean_accuracy, mean_precision, mean_recall, mean_f1);
fprintf('标准差\t%.2f\t\t%.4f\t\t%.4f\t\t%.4f\n\n', ...
        std_accuracy, std_precision, std_recall, std_f1);

%% ==================== 第六部分：每类性能统计 ====================
fprintf('========================================\n');
fprintf('各故障类型平均性能 (10折平均)\n');
fprintf('========================================\n\n');

num_classes = length(unique(Y_all));
class_precision_avg = zeros(num_classes, 1);
class_recall_avg = zeros(num_classes, 1);
class_f1_avg = zeros(num_classes, 1);

for i = 1:num_classes
    precision_sum = 0;
    recall_sum = 0;
    f1_sum = 0;
    
    for fold = 1:k_folds
        precision_sum = precision_sum + fold_results.precision{fold}(i);
        recall_sum = recall_sum + fold_results.recall{fold}(i);
        f1_sum = f1_sum + fold_results.f1_score{fold}(i);
    end
    
    class_precision_avg(i) = precision_sum / k_folds;
    class_recall_avg(i) = recall_sum / k_folds;
    class_f1_avg(i) = f1_sum / k_folds;
end

fprintf('故障类型\t精确率\t\t召回率\t\tF1分数\n');
fprintf('--------------------------------------------------------\n');
for i = 1:num_classes
    fprintf('y%d\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
            i, class_precision_avg(i), class_recall_avg(i), class_f1_avg(i));
end
fprintf('\n');

%% ==================== 第七部分：可视化结果 ====================
fprintf('生成可视化结果...\n');

figure('Position', [100, 100, 1400, 900]);

% 1. 每折准确率折线图
subplot(2, 3, 1);
plot(1:k_folds, fold_results.accuracy, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
yline(mean_accuracy, 'r--', 'LineWidth', 2);
xlabel('折数');
ylabel('准确率 (%)');
title('各折准确率');
legend('各折准确率', '平均准确率', 'Location', 'best');
grid on;
ylim([min(fold_results.accuracy)-5, 105]);

% 2. 性能指标对比柱状图
subplot(2, 3, 2);
metrics = [mean_precision, mean_recall, mean_f1, mean_accuracy/100];
bar(metrics, 'FaceColor', [0.3, 0.6, 0.8]);
set(gca, 'XTickLabel', {'精确率', '召回率', 'F1分数', '准确率'});
ylabel('分数');
title('平均性能指标');
ylim([0, 1.05]);
grid on;
for i = 1:length(metrics)
    text(i, metrics(i)+0.02, sprintf('%.4f', metrics(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% 3. 各类性能对比
subplot(2, 3, 3);
x_pos = 1:num_classes;
bar_width = 0.25;
bar(x_pos - bar_width, class_precision_avg, bar_width, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
bar(x_pos, class_recall_avg, bar_width, 'FaceColor', [0.8, 0.4, 0.2]);
bar(x_pos + bar_width, class_f1_avg, bar_width, 'FaceColor', [0.4, 0.8, 0.4]);
xlabel('故障类型');
ylabel('分数');
title('各类故障性能对比');
legend('精确率', '召回率', 'F1分数', 'Location', 'best');
set(gca, 'XTick', 1:num_classes, 'XTickLabel', ...
    arrayfun(@(x) sprintf('y%d', x), 1:num_classes, 'UniformOutput', false));
grid on;
ylim([0, 1.05]);

% 4. 混淆矩阵热图 (最后一折)
subplot(2, 3, 4);
imagesc(fold_results.confusion_matrix{k_folds});
colorbar;
colormap('jet');
xlabel('预测类别');
ylabel('真实类别');
title(sprintf('混淆矩阵 (第%d折)', k_folds));
set(gca, 'XTick', 1:num_classes, 'YTick', 1:num_classes);
for i = 1:num_classes
    for j = 1:num_classes
        text(j, i, num2str(fold_results.confusion_matrix{k_folds}(i,j)), ...
             'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
    end
end

% 5. 箱线图 - 显示各指标的分布
subplot(2, 3, 5);
boxplot([all_precision_means, all_recall_means, all_f1_means, fold_results.accuracy/100], ...
        'Labels', {'精确率', '召回率', 'F1分数', '准确率'});
ylabel('分数');
title('性能指标分布 (箱线图)');
grid on;
ylim([0, 1.05]);

% 6. 每折各指标对比雷达图风格
subplot(2, 3, 6);
plot(1:k_folds, all_precision_means, 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:k_folds, all_recall_means, 'g-s', 'LineWidth', 2, 'MarkerSize', 6);
plot(1:k_folds, all_f1_means, 'b-^', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('折数');
ylabel('分数');
title('各折指标变化趋势');
legend('精确率', '召回率', 'F1分数', 'Location', 'best');
grid on;
ylim([0, 1.05]);

% 保存图形
saveas(gcf, 'BP_SVM_10Fold_Results.png');
fprintf('  可视化图表已保存: BP_SVM_10Fold_Results.png\n\n');

%% ==================== 第八部分：保存结果 ====================
fprintf('保存交叉验证结果...\n');

save('BP_SVM_10Fold_Results.mat', 'fold_results', 'k_folds', ...
     'mean_accuracy', 'std_accuracy', 'mean_precision', 'std_precision', ...
     'mean_recall', 'std_recall', 'mean_f1', 'std_f1', ...
     'class_precision_avg', 'class_recall_avg', 'class_f1_avg');

fprintf('  结果已保存: BP_SVM_10Fold_Results.mat\n\n');

fprintf('========================================\n');
fprintf('10折交叉验证完成！\n');
fprintf('========================================\n');