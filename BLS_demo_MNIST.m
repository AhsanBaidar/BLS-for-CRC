clear all; clc;
load CCHI-2_FC6;

train_x = double(train_x/255);
train_y = double(train_y);
test_x = double(test_x/255);
test_y = double(test_y);
train_y=(train_y-1)*2+1;
test_y=(test_y-1)*2+1;
assert(isfloat(train_x), 'train_x must be a float');
assert(all(train_x(:)>=0) && all(train_x(:)<=1), 'all data in train_x must be in [0:1]');
assert(isfloat(test_x), 'test_x must be a float');
assert(all(test_x(:)>=0) && all(test_x(:)<=1), 'all data in test_x must be in [0:1]');

C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes
L=10;%feature nodes  per window
N=10;% number of windows of feature nodes
M=520;% number of enhancement nodes
epochs=1;% number of epochs 
train_err=zeros(1,epochs);test_err=zeros(1,epochs);
train_time=zeros(1,epochs);test_time=zeros(1,epochs);

for j=1:epochs    
    [TrainingAccuracy,TestingAccuracy,Training_time,Testing_time] = bls_train(train_x,train_y,test_x,test_y,s,C,L,N,M);       
    train_err(j)=TrainingAccuracy * 100;
    test_err(j)=TestingAccuracy * 100;
    train_time(j)=Training_time;
    test_time(j)=Testing_time;
end




