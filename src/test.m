clear;clc;
load('HR_92.mat');
correct_no = 0;
predict_class_set=zeros(280,1);
for i = 1 : test_no * 40
    dist_min = exp(1000);
    for j = 1 : train_no * 40
%         dist = norm(HR_test_PCA_features(:, i)-HR_PCA_features(:, j));
        dist = sum(abs(HR_test_PCA_features(:, i)-HR_PCA_features(:, j)));
        if dist < dist_min
            predict_class = str2double(char(HR_test_label(j)));
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