clc
clear
close all

%% Read DS
imds = imageDatastore('C:\Users\maher\OneDrive\Desktop\Coor paper\Coffe DS\DS 3 Code\DS3 Binary\1-Train Network\Coffee DB Binary', 'IncludeSubfolders', true,'LabelSource', 'foldernames');

%% Preprocess Fn
imds.ReadFcn = @(filename)preprocess_DB(filename);
%% Split DB
[trainingImages, testImages] = splitEachLabel(imds, 0.7, 'randomize');

%% Tramsfer Learning For AlexNet

net = alexnet; 
inputSize = net.Layers(1).InputSize
layers = net.Layers 
%%
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
%%

lgraph = removeLayers(lgraph, {'fc8','prob','output'});

%%
newLayers = [
    fullyConnectedLayer(2,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor', 10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];

%%
lgraph = addLayers(lgraph,newLayers);

%%
lgraph = connectLayers(lgraph,'drop7','fc');

%%
opts = trainingOptions('sgdm', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate', 0.001,'MiniBatchSize',20,'MaxEpochs',10,'Plots','training-progress');


%% Train the Network

Coeffe_alexnet = trainNetwork(trainingImages,lgraph, opts);
save('Coeffe_alexnet.mat','Coeffe_alexnet','-v7.3')

%% This is Accuarcy Using Normal CNN_AlexNet

predictedLabels = classify(Coeffe_alexnet, testImages); 
accuracy_alexnet = mean(predictedLabels == testImages.Labels)*100

save('accuracy_alexnet.mat','accuracy_alexnet','-v7.3')



%%
x=double(testImages.Labels);
y=double(predictedLabels);
[FP_rate,TP_rate,~,AUC]=perfcurve(x,y,2);
figure
plot(FP_rate,TP_rate,'b-');
title("ROC of ResNet18")
grid on
xlabel('False Positive Rate');
ylabel('True Positive Rate');
% Area under the ROC curve value
AUC
save('AUC.mat','AUC','-v7.3')
%%

figure
cm_LSTM = confusionchart(testImages.Labels,predictedLabels);
cm_LSTM.Title = 'Confusion Matrix of Renet18 Network';



