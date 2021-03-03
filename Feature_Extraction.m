imds = imageDatastore('Path','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');

load 'lastNet_TEXTURE_VGG'
net = myNet;
inputSize = net.Layers(1).InputSize;
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'fc6';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

Train_Y = imdsTrain.Labels;
Test_Y = imdsTest.Labels;

%%%%%%%Feature Modification %%%%%%%%%%
min_test =  min(min(featuresTest));
min_train = min(min(featuresTrain));

test_x_new = featuresTest-min_test;
train_x_new = featuresTrain-min_train;

max_test = max(max(test_x_new));
max_train = max(max(train_x_new));

test_x = test_x_new/max_test;
train_x= train_x_new/max_train;



%%%%%%%% Label Generation %%%%%%%%%

train_y = [];
for i = 1:length(Train_Y)
    if(Train_Y(i)== 'COM')
        train_y(i,:) = [1 0 0 0 0 0];
    elseif(Train_Y(i)=='DEB')
        train_y(i,:)=[0 1 0 0 0 0];
    elseif(Train_Y(i)=='LYM')
        train_y(i,:)=[0 0 1 0 0 0];
    elseif(Train_Y(i)=='MUC')
        train_y(i,:)=[0 0 0 1 0 0];
    elseif(Train_Y(i)=='STR')
        train_y(i,:)= [0 0 0 0 1 0];
    else
        train_y(i,:)= [0 0 0 0 0 1];
    end    
end

test_y = [];
for i = 1:length(Test_Y)
    if(Test_Y(i)== 'COM')
        test_y(i,:) = [1 0 0 0 0 0];
    elseif(Test_Y(i)=='DEB')
        test_y(i,:)=[0 1 0 0 0 0];
    elseif(Test_Y(i)=='LYM')
        test_y(i,:)=[0 0 1 0 0 0];
    elseif(Test_Y(i)=='MUC')
        test_y(i,:)=[0 0 0 1 0 0];
    elseif(Test_Y(i)=='STR')
        test_y(i,:)= [0 0 0 0 1 0];
    else
        test_y(i,:)= [0 0 0 0 0 1];
    end    
end

save('CCHI-1_FC6','train_x','train_y','test_x','test_y');