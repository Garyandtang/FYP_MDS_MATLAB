

%%%%%% Reshape and display %%%%%%
for i = 1 : 10
    figure
    display = reshape(HR_eigenface(:, i), [112, 92]);
    imagesc(display);
    colormap(gray);
end

%%%%%% validation calculate the HR_PCA_feature distance %%%%%%
result = zeros(test_no * 40, 2);
for i = 1 : test_no * 40
   result(i,1) = norm(HR_test_PCA_features(:,7)-HR_test_PCA_features(:,i));
end
for i = 1 : train_no * 40
    result(i,2) = norm(HR_test_PCA_features(:,7)-HR_PCA_features(:,i));
end

%%%%%% Reconstruct the image based on the meanface and eigenface %%%%%%
figure
re = zeros(1296, 1);
for i = 1 : 40
    re = re+ HR_test_PCA_features(i,1)*HR_eigenface(:, i);
end
re = re + HR_train_meanface;
display = reshape(re, [36, 36]);
imagesc(display);
colormap(gray);
figure
display2 = reshape(HR_test_data(:,1), [36, 36]);
imagesc(display2);
colormap(gray)


%%%%%% show eigenface by subplot %%%%%%
re = zeros(1296, 1);
for i = 1 : 20
    re = HR_eigenface(:, (i));
    display = reshape(re, [36, 36]);
    subplot(20/5,5,i);
    imagesc(display);
    colormap(gray);
end