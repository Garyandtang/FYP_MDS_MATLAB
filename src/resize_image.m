clear;
%parameter
HR_path = '/home/jiawei/Documents/polyU/FYP/Couple_mapping_MATLAB/test/HR/';
LR_path = '/home/jiawei/Documents/polyU/FYP/Couple_mapping_MATLAB/test/LR/';
LR_row = 12;
LR_colom = 12;
%create folder for LR images
if ~exist(LR_path)
    mkdir(LR_path);
end
%get the file name from HR folder
fileFolder=fullfile(HR_path);%文件夹名plane
dirOutput=dir(fullfile(fileFolder,'*.jpg'));%如果存在不同类型的文件，用‘*’读取所有，如果读取特定类型文件，'.'加上文件类型，例如用‘.jpg’
HR_image_name_all={dirOutput.name}'; 
I_size = size(HR_image_name_all);
I_size = I_size(1);

%resize and save
for i = 1 : I_size
    HR_image_name = strcat(HR_path,char(HR_image_name_all(i)));
    LR_image_name = strcat(LR_path,char(HR_image_name_all(i)));
    HR_image = imread(HR_image_name);
    LR_image = imresize(HR_image, [LR_row LR_colom], 'cubic');
    imwrite(LR_image, LR_image_name);
end
