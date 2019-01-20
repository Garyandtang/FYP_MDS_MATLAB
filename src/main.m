clc;
%Parameter
LR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/LR_12.txt';
HR_file = '/home/jiawei/Documents/polyU/FYP/MDS_MATLAB/data/att_faces/HR_36.txt';
train_no = 3;
NumOfEigenface = 10;

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
%%%%%% PCA feature %%%%%%
LR_train_meanface = get_meanface(LR_train_data);
HR_train_meanface = get_meanface(HR_train_data);

for i = 1: (train_no * 40)
    HR_DemeanFace(:,i) = HR_train_data(:,i) - HR_train_meanface;
    LR_DemeanFace(:,i) = LR_train_data(:,i) - LR_train_meanface;
end

HR_eigenface = get_eigenface(HR_DemeanFace);
LR_eigenface = get_eigenface(LR_DemeanFace);



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