clear;clc;
%%%%%% Parameter %%%%%%
LR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/LR_12.txt';
HR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/HR_36.txt';
train_no = 1;
test_no = 10 - train_no;
NumOfEigenface = 7;
landa = 0.5;
common_space_dim = 13;
iteration = 20;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% load DATA %%%%%%
[LR_train_path, LR_train_label, LR_test_path, LR_test_label] = get_path_label(LR_file, train_no);
[HR_train_path, HR_train_label, HR_test_path, HR_test_label] = get_path_label(HR_file, train_no);
for i = 1: (train_no * 40)
    LR_load = char(LR_train_path(i));
    HR_load = char(HR_train_path(i));
    LR_Image = imread(LR_load);
    HR_Image = imread(HR_load);
    LR_train_data(:,i) = double(reshape(LR_Image, [ ], 1));
    HR_train_data(:,i) = double(reshape(HR_Image, [ ], 1));
end

%%%%%% Get PCA features %%%%%%
LR_train_meanface = get_meanface(LR_train_data);
HR_train_meanface = get_meanface(HR_train_data);

for i = 1: (train_no * 40)
    HR_DemeanFace(:,i) = HR_train_data(:,i) - HR_train_meanface;
    LR_DemeanFace(:,i) = LR_train_data(:,i) - LR_train_meanface;
end
%%%%% get eigenface and PCA features %%%%%
HR_eigenface = get_eigenface(HR_DemeanFace);
LR_eigenface = get_eigenface(LR_DemeanFace);

HR_PCA_features = (HR_DemeanFace'*HR_eigenface(:,1:NumOfEigenface))';
LR_PCA_features = (LR_DemeanFace'*LR_eigenface(:,1:NumOfEigenface))';

% [HR_eigenface, H_order] = get_eigenface_v2(HR_DemeanFace);
% [LR_eigenface, L_order] = get_eigenface_v2(LR_DemeanFace);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% load test DATA %%%%%%
for i = 1: (test_no * 40)
    LR_load = char(LR_test_path(i));
    HR_load = char(HR_test_path(i));
    LR_Image = imread(LR_load);
    HR_Image = imread(HR_load);
    LR_test_data(:,i) = double(reshape(LR_Image, [ ], 1));
    HR_test_data(:,i) = double(reshape(HR_Image, [ ], 1));
end

%%%%%% Get PCA features %%%%%%

for i = 1: (test_no * 40)
    HR_test_DemeanFace(:,i) = HR_test_data(:,i) - HR_train_meanface;
    LR_test_DemeanFace(:,i) = LR_test_data(:,i) - LR_train_meanface;
end


%%%%%% Get Distance %%%%%%
predict_class_set = zeros(400, 1);
HR_test_PCA_features = (HR_test_DemeanFace'*HR_eigenface(:,1:NumOfEigenface))';
LR_test_PCA_features = (LR_test_DemeanFace'*LR_eigenface(:,1:NumOfEigenface))';
HR_test_PCA_features = HR_test_PCA_features/1000;
LR_test_PCA_features = LR_test_PCA_features/1000;
HR_PCA_features = HR_PCA_features/1000;
LR_PCA_features = LR_PCA_features/1000;
correct_no = 0;
for i = 1 : test_no * 40
    dist_min = exp(1000);
    for j = 1 : train_no * 40
        %dist = norm(HR_test_PCA_features(:, i)-HR_PCA_features(:, j));
        dist = sum(abs(HR_test_PCA_features(:, i)-HR_PCA_features(:, j)));
        if dist < dist_min
            predict_class = str2double(char(HR_train_label(j)));
            dist_min = dist;
        end
    end
    predict_class_set(i) = predict_class;
    true_class = str2double(char(HR_test_label(i)));
    if predict_class == true_class
        correct_no = correct_no + 1;
    end
    
end
recognition_rate = correct_no/(test_no*40)


% % %%%%%% Reconstruct %%%%%%
% figure
% re = zeros(1296, 1);
% for i = 1 : 40
%     re = re+ HR_test_PCA_features(i,1)*HR_eigenface(:, i);
% end
% re = re + HR_train_meanface;
% display = reshape(re, [36, 36]);
% imagesc(display);
% colormap(gray);
% figure
% display2 = reshape(HR_test_data(:,1), [36, 36]);
% imagesc(display2);
% colormap(gray)



%%%%%% validation calculate the HR_PCA_feature distance %%%%%%
result = zeros(test_no * 40, 2);
for i = 1 : test_no * 40
   result(i,1) = norm(HR_test_PCA_features(:,7)-HR_test_PCA_features(:,i));
end
for i = 1 : train_no * 40
    result(i,2) = norm(HR_test_PCA_features(:,7)-HR_PCA_features(:,i));
end
% correct_no = 0;
% for i = 1 : test_no * 40
%     dist_min = exp(1000);
%     for j = 1 : test_no * 40
% %         dist = norm(HR_test_PCA_features(:, i)-HR_PCA_features(:, j));
%         if i == j
%             dist = exp(1000);
%             dist = sum(abs(HR_test_PCA_features(:, i)-HR_test_PCA_features(:, j)));
%         else
%             dist = sum(abs(HR_test_PCA_features(:, i)-HR_test_PCA_features(:, j)));
%         end
%         if dist < dist_min
%             predict_class = str2double(char(HR_test_label(j)));
%             dist_min = dist;
%         end
%     end
%     predict_class_set(i) = predict_class;
%     true_class = str2double(char(HR_test_label(i)));
%     if predict_class == true_class
%         correct_no = correct_no + 1;
%     end   
% end
% recognition_rate = correct_no/(test_no*40)

% for i = 1 : 10
%     figure
%     display = reshape(HR_eigenface(:, i), [36, 36]);
%     imagesc(display);
%     colormap(gray);
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [EV, order] = get_eigenface_v2(data)
covFace = cov(data');
[EV, ED] = eig(covFace);
for i = 1 : length(ED)
    Eigenvalue(i) = ED(i,i);
end
Eigenvalue_sorted = sort(Eigenvalue,'descend');
for i = 1 : length(ED)
    for j = 1 : length(ED)
        if Eigenvalue_sorted(i) == Eigenvalue(j)
            order(i) = j;
        end
    end
end
end

function eigenface = get_eigenface(data)
covFace = cov(data');
[EV, ED] = eig(covFace);
for i = 1 : length(ED)
    Eigenvalue(i) = ED(i,i);
end
Eigenvalue_sorted = sort(Eigenvalue,'descend');
for i = 1 : length(ED)
    for j = 1 : length(ED)
        if Eigenvalue_sorted(i) == Eigenvalue(j)
            order(i) = j;
        end
    end
end
EV = EV';
eigenface = EV(order, :);
eigenface = eigenface';
end

function meanFace = get_meanface(data)
meanFace = zeros(size(data, 1),1);
for i = 1 : size(data, 2)
    meanFace = meanFace+data(:,i);
end
meanFace = meanFace/size(data, 2);
end

function[train_path, train_label, test_path, test_label] = get_path_label(the_file, n)
train_path = strings(n * 40,1);
train_label = strings(n * 40,1);
test_path = strings((10-n) * 40,1);
test_label = strings((10-n) * 40,1);
fid = fopen(the_file);
train_i = 1;
test_i = 1;
i = 1;
while feof(fid)~=1
    line = strsplit(fgetl(fid), ',');
%     fprintf(num2str(i));
%     fprintf('\n');
    if (mod(i, 10) < n)
        train_path(train_i) = line(1);
        train_label(train_i) = line(2);
        train_i = train_i + 1;
    else
        test_path(test_i) = line(1);
        test_label(test_i) = line(2);
        test_i = test_i + 1;
    end
    i = i + 1;
end
fclose(fid);
end
% %DISPLAY 
% figure;
% for i = 1: 20
%     Display = HR_test_eigenface(:,i);
%     Display = reshape(Display, [36 36]);
%     subplot(20/5,5,i);
%     imagesc(Display) 
%     colorbar
%     colormap(gray)
% end