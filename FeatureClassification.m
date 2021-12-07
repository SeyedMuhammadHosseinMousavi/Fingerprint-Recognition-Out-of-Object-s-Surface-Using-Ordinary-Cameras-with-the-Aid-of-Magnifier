%% 
% This code runs over 3 classes of finger prints, captured by fluorescent powder
% UV light plus in normal mode. System takes the input and after
% preprocessing, starts to extracting different features from diffrent domains.
% Then, lasso feature selection and classification by 5 types plus CNN. Also, 
% related plots are provided by the code. Data is collected for three
% subjects using macro lens plus fluorescent powder under uv light.
% You can play with parameters or change the dataset for any type of
% application related to image classification.
% Seyed Muhammad Hossein Mousavi - mosavi.a.i.buali@gmail.com

clear;clc;
%Read the input images
path='All';
fileinfo = dir(fullfile(path,'*.jpg'));
filesnumber=size(fileinfo);
for i = 1 : filesnumber(1,1)
images{i} = imread(fullfile(path,fileinfo(i).name));
    disp(['Loading image No :   ' num2str(i) ]);
end;
% Color to Gray Conversion
for i = 1 : filesnumber(1,1)
gray{i}=rgb2gray(images{i}); 
    disp(['To Gray :   ' num2str(i) ]);
end;
% Resize Images
% for i = 1 : filesnumber(1,1)
% resized{i}=imresize(gray{i}, [256 256]); 
%     disp(['Image Resized :   ' num2str(i) ]);
% end;
% Contrast Adjustment
for i = 1 : filesnumber(1,1)
adjusted{i}=imadjust(gray{i}); 
    disp(['Image Adjust :   ' num2str(i) ]);
end;
%%
% Extract SURF Features (just color)
imset = imageSet('AllSURF','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',10,'PointSelection','Detector');
% Encode the images as new features
surf = encode(bag,imset);

% Extract LPQ Features (just depth)
% LPQ
winsize=19;
for i = 1 : filesnumber(1,1)
tmp{i}=lpq(adjusted{i},winsize);
disp(['Extract LPQ :   ' num2str(i) ]);end;
for i = 1 : filesnumber(1,1)lpq(i,:)=tmp{i};end;

% Extract LBP Features  
for i = 1 : filesnumber(1,1)
    % The less cell size the more accuracy 
lbp{i} = extractLBPFeatures(adjusted{i},'CellSize',[64 64]);
    disp(['Extract LBP :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
    lbpfeature(i,:)=lbp{i};
    disp(['LBP To Matrix :   ' num2str(i) ]);
end;
% lasso feature selection
disp(['Working On Lasso For LBP (Please Wait) ...']);
lbpmain=lbpfeature;
% Labeling for lasso
clear lasso;clear B;clear Stats;clear ds;
label(1:60,1)=1;
label(61:120,1)=2;
label(121:180,1)=3;
% clear lasso;
[B Stats] = lasso(lbpfeature,label, 'CV', 5);
disp(B(:,1:5))
disp(Stats)
%
lassoPlot(B, Stats, 'PlotType', 'CV')
ds.Lasso = B(:,Stats.IndexMinMSE);
disp(ds)
sizemfcc=size(lbpfeature);
temp=1;       
for i=1:sizemfcc(1,2)
if ds.Lasso(i)~=0
lasso(:,temp)=lbpfeature(:,i);
temp=temp+1;end;end;lbpfeature=lasso;

% Extract HOG Features (just color)
for i = 1 : filesnumber(1,1)
    % The less cell size the more accuracy 
hog{i} = extractHOGFeatures(adjusted{i},'CellSize',[64 64]);
    disp(['Extract HOG :   ' num2str(i) ]);
end;
for i = 1 : filesnumber(1,1)
    hogfeature(i,:)=hog{i};
    disp(['HOG To Matrix :   ' num2str(i) ]);
end;
% lasso feature selection
disp(['Working On Lasso For LBP (Please Wait) ...']);
hogmain=hogfeature;
% Labeling for lasso
clear lasso;clear B;clear Stats;clear ds;
label(1:60,1)=1;
label(61:120,1)=2;
label(121:180,1)=3;
% clear lasso;
[B Stats] = lasso(hogfeature,label, 'CV', 5);
disp(B(:,1:5))
disp(Stats)
%
lassoPlot(B, Stats, 'PlotType', 'CV')
ds.Lasso = B(:,Stats.IndexMinMSE);
disp(ds)
sizemfcc=size(hogfeature);
temp=1;       
for i=1:sizemfcc(1,2)
if ds.Lasso(i)~=0
lasso(:,temp)=hogfeature(:,i);
temp=temp+1;end;end;hogfeature=lasso;

% Combining Feature Matrixes
FinalReady=[surf lpq lbpfeature hogfeature];
% Labels
sizefinal=size(FinalReady);
sizefinal=sizefinal(1,2);
FinalReady(1:60,sizefinal+1)=1;
FinalReady(61:120,sizefinal+1)=2;
FinalReady(121:180,sizefinal+1)=3;
%%
% Classification
% KNN 
lblknn=FinalReady(:,end);
dataknn=FinalReady(:,1:end-1);
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',5,'Standardize',1)
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat)
KNNAccuracy = 1 - kfoldLoss(knndat, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedknn = resubPredict(Mdl);
% Plot Confusion Matrix
figure
cmknn = confusionchart(lblknn,predictedknn);
cmknn.Title = 'KNN';
cmknn.RowSummary = 'row-normalized';
cmknn.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoreknn] = resubPredict(Mdl);
diffscoreknn = scoreknn(:,2) - max(scoreknn(:,1),scoreknn(:,3));
[Xknn,Yknn,T,~,OPTROCPTknn,suby,subnames] = perfcurve(lblknn,diffscoreknn,1);
%
figure;
plot(Xknn,Yknn)
hold on
plot(OPTROCPTknn(1),OPTROCPTknn(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for KNN')
hold off
%
knnsss=size(Xknn);knnsss=knnsss(1,1);
mx=min(Xknn(Xknn>0));
Preknn=max(Xknn)-mx;
my=min(Yknn(Yknn>0));
Recknn=max(Yknn)-my;
disp(['K-NN Precision :   ' num2str(Preknn) ]);
disp(['K-NN Recall :   ' num2str(Recknn) ]);

% SVM
svmclass = fitcecoc(dataknn,lblknn);
svmerror = resubLoss(svmclass);
CVMdl = crossval(svmclass);
genError = kfoldLoss(CVMdl);
% Compute validation accuracy
SVMAccuracy = 1 - kfoldLoss(CVMdl, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedsvm = resubPredict(svmclass);
% Plot Confusion Matrix
figure
cmsvm = confusionchart(lblknn,predictedsvm);
cmsvm.Title = 'SVM';
cmsvm.RowSummary = 'row-normalized';
cmsvm.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoresvm] = resubPredict(svmclass);
diffscoresvm = scoresvm(:,2) - max(scoresvm(:,1),scoresvm(:,3));
[Xsvm,Ysvm,T,~,OPTROCPTsvm,suby,subnames] = perfcurve(lblknn,diffscoresvm,1);
%
figure;
plot(Xsvm,Ysvm)
hold on
plot(OPTROCPTsvm(1),OPTROCPTsvm(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for SVM')
hold off
%
svmsss=size(Xsvm);
svmsss=svmsss(1,1);
mx=min(Xsvm(Xsvm>0));
my=min(Ysvm(Ysvm>0));
Presvm=max(Xsvm)-mx;
Recsvm=max(Ysvm)-my;
disp(['SVM Precision :   ' num2str(Presvm) ]);
disp(['SVM Recall :   ' num2str(Recsvm) ]);

% LDA
MdlLinear = fitcdiscr(dataknn,lblknn);
Lin = resubLoss(MdlLinear);
ldadat = crossval(MdlLinear);
LDAAccuracy = 1 - kfoldLoss(ldadat, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedlda = resubPredict(MdlLinear);
% Plot Confusion Matrix
figure
cmlda = confusionchart(lblknn,predictedlda);
cmlda.Title = 'LDA';
cmlda.RowSummary = 'row-normalized';
cmlda.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scorelda] = resubPredict(MdlLinear);
diffscorelda = scorelda(:,2) - max(scorelda(:,1),scorelda(:,3));
[Xlda,Ylda,T,~,OPTROCPTlda,suby,subnames] = perfcurve(lblknn,diffscorelda,3);
%
figure;
plot(Xlda,Ylda)
hold on
plot(OPTROCPTlda(1),OPTROCPTlda(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for LDA')
hold off
%
ldasss=size(Xlda);
ldasss=ldasss(1,1);
mx=min(Xlda(Xlda>0));
my=min(Ylda(Ylda>0));
Prelda=max(Xlda)-mx;
Reclda=max(Ylda)-my;
disp(['LDA Precision :   ' num2str(Prelda) ]);
disp(['LDA Recall :   ' num2str(Reclda) ]);

% Ensemble Subspace Discriminant
disc = templateDiscriminant('DiscrimType','pseudoLinear');
Mdlens = fitcensemble(dataknn,lblknn,'Method','Subspace','Learners',disc);
SDLE = resubLoss(Mdl);
ensdat = crossval(Mdlens);
ENSAccuracy = 1 - kfoldLoss(ensdat, 'LossFun', 'ClassifError');
% Predict the labels of the training data.
predictedens = resubPredict(Mdlens);
% Plot Confusion Matrix
figure
cmens = confusionchart(lblknn,predictedens);
cmens.Title = 'Ensemble';
cmens.RowSummary = 'row-normalized';
cmens.ColumnSummary = 'column-normalized';
% Precision, Recall and ROC
[~,scoreens] = resubPredict(Mdlens);
diffscoreens = scoreens(:,2) - max(scoreens(:,1),scoreens(:,3));
[Xens,Yens,T,~,OPTROCPTens,suby,subnames] = perfcurve(lblknn,diffscoreens,3);
%
figure;
plot(Xens,Yens)
hold on
plot(OPTROCPTens(1),OPTROCPTens(2),'ro')
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC Curve for Ensemble')
hold off
%
enssss=size(Xens);
enssss=enssss(1,1);
mx=min(Xens(Xens>0));
my=min(Yens(Yens>0));
Preens=max(Xens)-mx;
Recens=max(Yens)-my;
disp(['Ensemble Precision :   ' num2str(Preens) ]);
disp(['Ensemble Recall :   ' num2str(Recens) ]);

% Shallow Neural Network
network=FinalReady(:,1:end-1);
netlbl=FinalReady(:,end);
sizenet=size(network);
sizenet=sizenet(1,1);
for i=1 : sizenet
            if netlbl(i) == 1
               netlbl2(i,1)=1;
        elseif netlbl(i) == 2
               netlbl2(i,2)=1;
        elseif netlbl(i) == 3
               netlbl2(i,3)=1; 
        end
end
% Changing data shape from rows to columns
network=network'; 
% Changing data shape from rows to columns
netlbl2=netlbl2'; 
% Defining input and target variables
inputs = network;
targets = netlbl2;
% Create a Pattern Recognition Network
hiddenLayerSize = 100;
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
% Polak-Ribiére Conjugate Gradient
net = feedforwardnet(10, 'traincgp');
%
[net,tr] = train(net,inputs,targets);
% Test the Network
outputs = net(inputs);
%
errors = gsubtract(targets,outputs);
%
performance = perform(net,targets,outputs);
% Polak-Ribiére Conjugate Gradient
figure, plottrainstate(tr)
% Plot Confusion Matrixes
figure, plotconfusion(targets,outputs);
title('Polak-Ribiére Conjugate Gradient');
% Res
disp(['K-NN Classification Accuracy :   ' num2str(KNNAccuracy*100) ]);
disp(['SVM Classification Accuracy :   ' num2str(SVMAccuracy*100) ]);
disp(['LDA Classification Accuracy :   ' num2str(LDAAccuracy*100) ]);
disp(['Ensemble Classification Accuracy :   ' num2str(ENSAccuracy*100) ]);
disp(['Shallow NN Classification Accuracy :   ' num2str(100-performance) ]);

%% CNN  
deepDatasetPath = fullfile('AllCNN');
imds = imageDatastore(deepDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% Number of training (less than number of each class)
numTrainFiles = 110;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
layers = [
    % Input image size for instance: 512 512 3
    imageInputLayer([256 128 3])
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    % Number of classes
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',80, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',7, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation) *100;
disp(['CNN Recognition Accuracy Is =   ' num2str(accuracy) ]);

% Demo Sample for Different Data
pic=imread('non.jpg');
picg=rgb2gray(pic);
pich = histeq(picg);
picm = wiener2(pich,[3 3]);
pics = imsharpen(picm,'Radius',5,'Amount',0.5);
se = strel('disk',1);
eroded = imerode(pics,se);
%
pic1=imread('Mag.jpg');
picg1=rgb2gray(pic1);
pich1 = histeq(picg1);
picm1 = wiener2(pich1,[3 3]);
pics1 = imsharpen(picm1,'Radius',5,'Amount',0.5);
se = strel('disk',1);
eroded1 = imerode(pics1,se);
%
pic2=imread('MagUV.jpg');
picg2=rgb2gray(pic2);
pich2 = histeq(picg2);
picm2 = wiener2(pich2,[3 3]);
pics2 = imsharpen(picm2,'Radius',5,'Amount',0.5);
se = strel('disk',1);
eroded2 = imerode(pics2,se);
%
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,4,1),imshow(pic);
title('Raw Original');
subplot(3,4,2),imshow(picg);
title('Raw Gray');
subplot(3,4,3),imshow(picm);
title('Raw Enhanced');
subplot(3,4,4),imshow(eroded);
title('Raw Erosion');
subplot(3,4,5),imshow(pic1);
title('Magnifier Original');
subplot(3,4,6),imshow(picg1);
title('Magnifier Gray');
subplot(3,4,7),imshow(picm1);
title('Magnifier Enhanced');
subplot(3,4,8),imshow(eroded1);
title('Magnifier Erosion');
subplot(3,4,9),imshow(pic2);
title('UV Macro Original');
subplot(3,4,10),imshow(picg2);
title('UV Macro Gray');
subplot(3,4,11),imshow(picm2);
title('UV Macro Enhanced');
subplot(3,4,12),imshow(eroded2);
title('UV Macro Erosion');
figure
montage({picm, picm1, picm2});



