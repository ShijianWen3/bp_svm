%% BP-SVM采煤机故障诊断 - 数据生成脚本
% 根据论文表1和表2生成模拟故障数据集

clear; clc; close all;

%% 1. 定义故障类型和特征
% 故障现象特征（6维输入）
% x1: 泵压力异常
% x2: 泵冷却压力异常  
% x3: 泵工作转速异常
% x4: 电动机温度异常
% x5: 电动机电流异常
% x6: 电动机转速异常

% 故障原因标签（8类输出）
% y1: 液压泵故障
% y2: 液压缸故障
% y3: 液压阀故障
% y4: 电动机故障
% y5: 电缆故障
% y6: 控制系统故障
% y7: 截割部故障
% y8: 牵引部故障

%% 2. 根据论文表1定义故障规则
% 故障现象 -> 故障原因映射关系
fault_rules = {
    % [x1, x2, x3, x4, x5, x6] -> 可能的故障类型
    [1, 0, 0, 0, 0, 0], [1, 2, 3, 6];  % x1异常 -> y1,y2,y3,y6
    [0, 1, 0, 0, 0, 0], [3, 6];        % x2异常 -> y3,y6
    [0, 0, 1, 0, 0, 0], [1, 6];        % x3异常 -> y1,y6
    [0, 0, 0, 1, 0, 0], [4, 6];        % x4异常 -> y4,y6
    [0, 0, 0, 0, 1, 0], 5;           % x5异常 -> y5
    [0, 0, 0, 0, 0, 1], [4, 5, 6, 7, 8]; % x6异常 -> y4,y5,y6,y7,y8
};

%% 3. 生成样本数据（按表2的分布）
% x1:60组, x2:26组, x3:25组, x4:33组, x5:16组, x6:45组
sample_counts = [60, 26, 25, 33, 16, 45];
total_samples = sum(sample_counts);

% 初始化数据矩阵
X_data = zeros(total_samples, 6);  % 特征矩阵
Y_data = zeros(total_samples, 1);  % 标签向量

% 设置随机种子，保证可重复性
rng(42);

%% 4. 为每种故障现象生成数据
sample_idx = 1;

for fault_type = 1:6
    num_samples = sample_counts(fault_type);
    
    for i = 1:num_samples
        % 基础特征模式（主要异常特征）
        base_feature = fault_rules{fault_type, 1};
        
        % 添加随机噪声，模拟真实测量
        % 异常特征：0.7-1.0范围（高值）
        % 正常特征：0.0-0.3范围（低值）
        feature_vector = zeros(1, 6);
        for j = 1:6
            if base_feature(j) == 1
                % 异常特征：高值 + 噪声
                feature_vector(j) = 0.7 + 0.3*rand() + 0.05*randn();
            else
                % 正常特征：低值 + 噪声
                feature_vector(j) = 0.15*rand() + 0.02*randn();
            end
        end
        
        % 添加一些耦合效应（故障间的相互影响）
        if fault_type == 1  % x1异常，可能影响x3
            feature_vector(3) = feature_vector(3) + 0.2*rand();
        elseif fault_type == 4  % x4异常，可能影响x6
            feature_vector(6) = feature_vector(6) + 0.15*rand();
        end
        
        % 裁剪到[0,1]范围
        feature_vector = max(0, min(1, feature_vector));
        
        % 根据规则随机选择一个故障原因
        possible_faults = fault_rules{fault_type, 2};
        fault_label = possible_faults(randi(length(possible_faults)));
        
        % 存储数据
        X_data(sample_idx, :) = feature_vector;
        Y_data(sample_idx) = fault_label;
        sample_idx = sample_idx + 1;
    end
end

%% 5. 打乱数据顺序
shuffle_idx = randperm(total_samples);
X_data = X_data(shuffle_idx, :);
Y_data = Y_data(shuffle_idx);

%% 6. 划分训练集和测试集（140:65）
train_ratio = 140/205;
train_size = 140;
test_size = 65;

X_train = X_data(1:train_size, :);
Y_train = Y_data(1:train_size);
X_test = X_data(train_size+1:end, :);
Y_test = Y_data(train_size+1:end);

%% 7. 数据归一化（Min-Max归一化）
X_min = min(X_train, [], 1);
X_max = max(X_train, [], 1);

% 避免除零
X_range = X_max - X_min;
X_range(X_range == 0) = 1;

X_train_norm = (X_train - X_min) ./ X_range;
X_test_norm = (X_test - X_min) ./ X_range;

%% 8. 保存数据
save('shearer_fault_data.mat', 'X_train', 'Y_train', 'X_test', 'Y_test', ...
     'X_train_norm', 'Y_train_norm', 'X_test_norm', 'Y_test_norm', ...
     'X_min', 'X_max', 'sample_counts');

%% 9. 数据可视化
figure('Position', [100, 100, 1200, 800]);

% 9.1 故障类型分布
subplot(2, 3, 1);
histogram(Y_data, 'BinMethod', 'integers', 'FaceColor', [0.2, 0.6, 0.8]);
xlabel('故障类型');
ylabel('样本数量');
title('故障类型分布');
grid on;

% 9.2 故障现象分布
subplot(2, 3, 2);
bar(sample_counts);
xlabel('故障现象 (x1-x6)');
ylabel('样本数量');
title('故障现象分布（表2）');
set(gca, 'XTickLabel', {'x1', 'x2', 'x3', 'x4', 'x5', 'x6'});
grid on;

% 9.3 特征相关性热图
subplot(2, 3, 3);
imagesc(corr(X_data));
colorbar;
xlabel('特征维度');
ylabel('特征维度');
title('特征相关性矩阵');
set(gca, 'XTick', 1:6, 'YTick', 1:6);

% 9.4 训练集特征分布（箱线图）
subplot(2, 3, 4);
boxplot(X_train_norm, 'Labels', {'x1', 'x2', 'x3', 'x4', 'x5', 'x6'});
ylabel('归一化特征值');
title('训练集特征分布（归一化后）');
grid on;

% 9.5 前两个主成分可视化
subplot(2, 3, 5);
[coeff, score] = pca(X_train_norm);
gscatter(score(:,1), score(:,2), Y_train);
xlabel('第一主成分');
ylabel('第二主成分');
title('PCA降维可视化（训练集）');
legend('Location', 'best');
grid on;

% 9.6 数据集统计信息
subplot(2, 3, 6);
axis off;
stats_text = {
    '数据集统计信息：';
    '─────────────────';
    sprintf('总样本数: %d', total_samples);
    sprintf('训练集: %d (%.1f%%)', train_size, train_ratio*100);
    sprintf('测试集: %d (%.1f%%)', test_size, (1-train_ratio)*100);
    '';
    '故障类型分布：';
    sprintf('y1-液压泵: %d', sum(Y_data==1));
    sprintf('y2-液压缸: %d', sum(Y_data==2));
    sprintf('y3-液压阀: %d', sum(Y_data==3));
    sprintf('y4-电动机: %d', sum(Y_data==4));
    sprintf('y5-电缆: %d', sum(Y_data==5));
    sprintf('y6-控制系统: %d', sum(Y_data==6));
    sprintf('y7-截割部: %d', sum(Y_data==7));
    sprintf('y8-牵引部: %d', sum(Y_data==8));
};
text(0.1, 0.9, stats_text, 'FontSize', 10, 'VerticalAlignment', 'top');

%% 10. 输出数据信息
fprintf('===== 数据生成完成 =====\n');
fprintf('总样本数: %d\n', total_samples);
fprintf('训练集: %d组\n', train_size);
fprintf('测试集: %d组\n', test_size);
fprintf('特征维度: 6\n');
fprintf('故障类型: 8类\n');
fprintf('\n各故障类型样本数:\n');
for i = 1:8
    fprintf('  y%d: %d组 (%.1f%%)\n', i, sum(Y_data==i), sum(Y_data==i)/total_samples*100);
end
fprintf('\n数据已保存至: shearer_fault_data.mat\n');
fprintf('========================\n');

%% 11. 显示部分样本数据
fprintf('\n前10组训练样本示例:\n');
fprintf('编号\t x1\t x2\t x3\t x4\t x5\t x6\t 故障类型\n');
fprintf('------------------------------------------------------------\n');
for i = 1:min(10, train_size)
    fprintf('%3d\t', i);
    fprintf('%.3f\t', X_train(i,:));
    fprintf('y%d\n', Y_train(i));
end