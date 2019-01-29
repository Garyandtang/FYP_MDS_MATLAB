clear;clc;
%%%%%% Parameter %%%%%%
LR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/LR_12.txt';
HR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/HR_36.txt';
train_no = 3;
test_no = 10 - train_no;
NumOfEigenface = 7;
landa = 0.5;
common_space_dim = 40;
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

HR_eigenface = get_eigenface(HR_DemeanFace);
LR_eigenface = get_eigenface(LR_DemeanFace);

HR_PCA_features = (HR_train_data'*HR_eigenface(:,1:NumOfEigenface))';
LR_PCA_features = (LR_train_data'*LR_eigenface(:,1:NumOfEigenface))';

%%%%%% Learn Tranformation Matrix %%%%%%
%%% construct new feature vector %%%
padding_zeros = zeros(NumOfEigenface, 120);
HR_PCA_features = [padding_zeros; HR_PCA_features];
LR_PCA_features = [LR_PCA_features; padding_zeros];

%%% obtain W %%%
W = -1 + 2*rand(2 * NumOfEigenface, common_space_dim);
ind = 0;
for i = 1 : iteration
    ind = ind + 1;
    fprintf(num2str(ind));
    fprintf('\n');
    V = W;
    A = get_A(LR_PCA_features, HR_PCA_features, LR_train_label, HR_train_label, landa);
    C = get_C(LR_PCA_features, HR_PCA_features, V, landa);
    W = pinv(A) * C * V;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
LR_test_meanface = get_meanface(LR_test_data);
HR_test_meanface = get_meanface(HR_test_data);

for i = 1: (train_no * 40)
    HR_DemeanFace_test(:,i) = HR_test_data(:,i) - HR_test_meanface;
    LR_DemeanFace_test(:,i) = LR_test_data(:,i) - LR_test_meanface;
end

HR_test_eigenface = get_eigenface(HR_DemeanFace_test);
LR_test_eigenface = get_eigenface(LR_DemeanFace_test);

HR_test_PCA_features = (HR_test_data'*HR_test_eigenface(:,1:NumOfEigenface))';
LR_test_PCA_features = (LR_test_data'*LR_test_eigenface(:,1:NumOfEigenface))';

%%%%%% transform to common space %%%%%%
W_LR = W(1:7, :);
W_HR = W(8:14, :);
HR_test_mapped = W_LR' * HR_test_PCA_features; % 15 * 280
LR_test_mapped = W_HR' * LR_test_PCA_features; % 15 * 280

%%%%%% calculate the recognition rate %%%%%%
correct_no = 0;
for i = 1 : train_no * 40
    dist_min = exp(1000);
    for j = 1 : train_no * 40
        dist = norm(LR_test_mapped(:, i)-LR_test_mapped(:, j));
        if dist < dist_min
            predict_class = str2double(char(LR_test_label(j)));
            dist_min = dist;
        end
    end
    true_class = str2double(char(LR_test_label(i)));
    if predict_class == true_class
        correct_no = correct_no + 1;
    end
    
end
recognition_rate = correct_no/(test_no*40)
% correct_no = 0;
% for i = 1 : train_no * 40
%     dist_min = exp(1000);
%     for j = 1 : train_no * 40
%         dist = norm(LR_test_PCA_features(:, i)-HR_test_PCA_features(:, j));
%         if dist < dist_min
%             predict_class = str2double(char(HR_test_label(j)));
%             dist_min = dist;
%         end
%     end
%     true_class = str2double(char(LR_test_label(i)));
%     if predict_class == true_class
%         correct_no = correct_no + 1;
%     end
%     
% end
% recognition_rate = correct_no/(test_no*40)
% save('S.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function C = get_C(LR_data, HR_data, V, landa)
LR_size = size(LR_data, 2);
HR_size = size(HR_data, 2);
C_size = size(HR_data, 1);
C = zeros(C_size);
ind = 0;
for i = 1:LR_size
    for j = 1:HR_size
%         ind = ind + 1;
%         fprintf(num2str(ind));
%         fprintf('\n');
        diff = LR_data(:,i) - HR_data(:,j);
        q = norm(V' * diff);
        d = norm(HR_data(:, i) - HR_data(:, j));
        if q > 0
            c = landa * d / q;
        else
            c = 0;
        end
        C = c * (diff * diff') + C;
    end
end

end
function A = get_A(LR_data, HR_data, LR_label, HR_label, landa)
LR_size = size(LR_data, 2);
HR_size = size(HR_data, 2);
A_size = size(HR_data, 1);
A = zeros(A_size);
for i = 1:LR_size
    for j = 1:HR_size
        diff = LR_data(:,i) - HR_data(:,j);
        if LR_label(i) == HR_label
            a = (1 - landa) + landa;
            A = a * (diff * diff') + A;
        else
            a = landa;
            A = a * (diff * diff') + A;
        end
    end
end

end

%%%remark I am not sure the distance function is correct %%%

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
    if (mod(i, 10) < 3)
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


% Display = reshape(LR_eigenface(:,1), [12 12]);
% imagesc(Display);
% colormap(gray);

% %DISPLAY 
% figure;
% for i = 1: 20
%     Display = HR_DemeanFace(:,i);
%     Display = reshape(Display, [36 36]);
%     subplot(20/5,5,i);
%     imagesc(Display) 
%     colorbar
%     colormap(gray)
% end