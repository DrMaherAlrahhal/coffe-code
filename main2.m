clc
clear
close all
%%
load lab
load features_all_efficientnetb0
load features_all_googlenet

all_features=[features_all_efficientnetb0 features_all_googlenet];
save('all_features.mat','all_features','-v7.3')
