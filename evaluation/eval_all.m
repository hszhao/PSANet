close all; clc; clear;
addpath('../PSPNet/matlab'); %add matcaffe path
addpath('visualizationCode');
data_name = 'ade20k'; %set to 'voc2012' or 'cityscapes' for relevant datasets

switch data_name
    case 'ade20k' %gt labels mapped from original 0-150 to 0-149 and 255 (ignore).
        isVal = true; %evaluation on valset
        data_root = '/mnt/sda1/hszhao/dataset/ade20k'; %root path of dataset
        eval_list = 'list/ade20k/ade20k_val.txt'; %evaluation list, refer to lists in folder 'samplelist'
        save_root = 'mc_result/ade20k/val/ss/psanet50_465/'; %root path to store the result image
        model_weights = 'model/ade20k/psanet50_ade20k_a8e884.caffemodel';
        model_deploy = 'prototxt/ade20k/psanet50_ade20k_465.prototxt';
        fea_cha = 150; %number of classes
        base_size = 512; %based size for scaling
        crop_size = 465; %crop size fed into network
        data_class = 'ade20k/ade20knames.mat'; %class name
        data_colormap = 'ade20k/ade20kcolors.mat'; %color map
    case 'voc2012' %gt labels are 0-20 and 255 (ignore).
        isVal = false; %evaluation on testset
        data_root = '/mnt/sda1/hszhao/dataset/voc2012';
        eval_list = 'list/voc2012/voc2012_test.txt';
        save_root = 'mc_result/voc2012/test/ms/psanet101_465/'; %with multiscale testing
        model_weights = 'model/voc2012/psanet101_voc2012_3c6a69.caffemodel';
        model_deploy = 'prototxt/voc2012/psanet101_voc2012_465.prototxt';
        fea_cha = 21;
        base_size = 512;
        crop_size = 465;
        data_class = 'voc2012/voc2012names.mat';
        data_colormap = 'voc2012/voc2012colors.mat';
    case 'cityscapes' %gt labels are 0-18 and 255 (ignore).
        isVal = true;
        data_root = '/mnt/sda1/hszhao/cityscapes';
        eval_list = 'list/cityscapes/cityscapes_val.txt';
        save_root = 'mc_result/cityscapes/val/ss/psanet101_705/';
        model_weights = 'model/cityscapes/psanet101_cityscapes_3ac1bf.caffemodel';
        model_deploy = 'prototxt/cityscapes/psanet101_cityscapes_705.prototxt';
        fea_cha = 19;
        base_size = 2048;
        crop_size = 705;
        data_class = 'cityscapes/cityscapesnames.mat';
        data_colormap = 'cityscapes/cityscapescolors.mat';
end
skipsize = 0; %skip several images in front of the list (in case of broken of last evaluation process)

is_save_feat = false; %set to true if final feature map is needed (not suggested for storage consuming)
save_gray_folder = [save_root 'gray/']; %path for predicted gray image
save_color_folder = [save_root 'color/']; %path for predicted color image
save_feat_folder = [save_root 'feat/']; %path for predicted feature map
scale_array = [1]; %set to [0.5 0.75 1 1.25 1.5 1.75] for multi-scale testing
mean_rgb = [123.68, 116.779, 103.939]; %means to be subtracted and the given values are used in our training stage

acc = double.empty;
iou = double.empty;
gpu_id = 0;

eval_sub(data_name,data_root,eval_list,model_weights,model_deploy,fea_cha,base_size,crop_size,data_class,data_colormap, ...
           is_save_feat,save_gray_folder,save_color_folder,save_feat_folder,gpu_id,skipsize,scale_array,mean_rgb);
if(isVal)
   eval_acc(data_name,data_root,eval_list,save_gray_folder,data_class,fea_cha);
end
