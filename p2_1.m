%% Machine Learning Project part 2: Classification %%

% clean up workspace
clear;
clc;

%% load sets and setup matrices
load ('training.csv');
load ('validation.csv');
load ('testing.csv');

numSamples = size(training,1);

%% iterate over cross-validation sets, different kernels
cvo = cvpartition(numSamples,'kfold',10);
error = [];
minerr = 100;

for i=1:cvo.NumTestSets
    % setup CV indices
    trIdx = cvo.training(i);
    teIdx = cvo.test(i);
    trSet = training(trIdx,:);
    cvSet = training(teIdx,:);
    
    %% train an SVM with RBF kernel on CV test sets
    SVMstruct = svmtrain(trSet(:,1:(end-1)),trSet(:,end),'kernel_function','rbf');
    eval = svmclassify(SVMstruct,cvSet(:,1:(end-1)));
    
    curr_err = geterror(eval,cvSet(:,end));
    
    % error vector for all CV sets
    error = [error curr_err];
    
    % save the minimal error set
    if(curr_err < minerr)
        minclass = eval;
        bestSVM = SVMstruct;
        minerr = curr_err;
    end
    
end

% debug output
error
minerr

% apply the best SVM to validation set
valid = svmclassify(bestSVM,validation);
csvwrite('classified_validation.csv', valid);
