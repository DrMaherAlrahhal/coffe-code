clc
clear
close all

%% Read DS
imds = imageDatastore('C:\Users\maher\OneDrive\Desktop\Coor paper\1-Coffe DS\DS 3 Code\DS3 Binary\1-Train Network\Coffee DB Binary', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

%% Preprocess Fn
imds.ReadFcn = @(filename)preprocess_DB(filename);
%% Split DB
[trainingImages, testImages] = splitEachLabel(imds, 0.7, 'randomize');

%% Tramsfer Learning For AlexNet

net = efficientnetb0; 
inputSize = net.Layers(1).InputSize
layers = net.Layers 
%%
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
%%

lgraph = removeLayers(lgraph, {'efficientnet-b0|model|head|dense|MatMul','Softmax','classification'});

%%
newLayers = [
    fullyConnectedLayer(2,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

%%
lgraph = addLayers(lgraph,newLayers);

%%
lgraph = connectLayers(lgraph,'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool','fc');

%%
opts = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate', 0.001,'MiniBatchSize',20,'MaxEpochs',10,'Plots','training-progress');


%% Train the Network

Coeffe_efficientnetb0= trainNetwork(trainingImages,lgraph, opts);
save('Coeffe_efficientnetb0.mat','Coeffe_efficientnetb0','-v7.3')

%% This is Accuarcy Using Normal CNN_AlexNet

predictedLabels = classify(Coeffe_efficientnetb0, testImages); 
accuracy_efficientnetb0 = mean(predictedLabels == testImages.Labels)*100

save('accuracy_efficientnetb0.mat','accuracy_efficientnetb0','-v7.3')



%%
x=double(testImages.Labels);
y=double(predictedLabels);
[FP_rate,TP_rate,~,AUC]=perfcurve(x,y,2);
figure
plot(FP_rate,TP_rate);
title("ROC of efficientnetb0")
grid on
xlabel('False Positive Rate');
ylabel('True Positive Rate');
% Area under the ROC curve value
AUC
save('AUC.mot','AUC','-v7.3')
%%

figure
cm_LSTM = confusionchart(testImages.Labels,predictedLabels);
cm_LSTM.Title = 'Confusion Matrix of efficientnetb0 Network';



