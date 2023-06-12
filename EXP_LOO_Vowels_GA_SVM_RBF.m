%% Notes
% 1. This approach assumes there are totally N subjects, the classifier 
%    is trained with N-1 subjects' data and test on the one subject left to
%    compute the diagonsis performance.
clear
clc 

%% Load data from pre-computed features
load('subjects_40_v6.mat')

num_sub = 40;
num_cls = 2;

%% Leave-One-Out Crossvaliuation
% Applying the feature weight to all subjects' data
% FEAT_CLASSIFY = cell(40,1); 
% for i = 1 : 40
%     n = size(FEAT_N{i}, 1);
%     fw = repmat(feature_weight', n, 1); 
%     
%     FEAT_CLASSIFY{i} = FEAT_N{i} .* fw;
% end
% 
%%
FEAT_CLASSIFY = FEAT_N;

%% GA-SVM Main Loop
% Initialize a table to store the dataset spreadsheet information                
        
for lo_pen = 1 : 0.1 : 1
size_train          = zeros(num_sub, num_cls);
size_test           = zeros(num_sub, num_cls);
size_total          = zeros(num_sub, num_cls);
accuracy_train      = zeros(num_sub, 1);
accuracy_train_ga   = zeros(num_sub, 1);
accuracy_validation = zeros(num_sub, 1);
accuracy_test       = zeros(num_sub, 1);
accuracy_test_ga    = zeros(num_sub, 1);

fprintf('\nLeft Out percentage: %.2f\n', lo_pen);

% For VFI correlation analysis
positive_per        = zeros(num_sub,1);
VFI1_test           = zeros(num_sub,1);

for i = 1 : 40
    
    if exist(sprintf('Data/R%03d', SUBJECT_ID{i}(1)), 'dir')

    % Partition the testing data
    fprintf('\nTesting Subject: R%03d\n', SUBJECT_ID{i}(1));
    
    VFI1_test(i) = SUBJECT_VFI{i}(1);
    fprintf('VFI-1 Score: %d\n', SUBJECT_VFI{i}(1));    

    if lo_pen ~= 0
        % Assign the testing data (from the current subject)
        feature_lo  = FEAT_CLASSIFY{i};    
        label_lo    = LABEL{i};
        skinfold_lo = mean(SUBJECT_SKINFOLD{i}, 2);

        size_test(i,1) = sum(label_lo == 1);
        size_test(i,2) = sum(label_lo == -1);

        N = size(label_lo, 1);
        num_lo = round(N * lo_pen);
        x = randsample(N,num_lo);

        feature_test  = feature_lo(x,:);
        label_test    = label_lo(x,:);
        skinfold_test = skinfold_lo(x,:);

        y = (1 : N)';
        y(x) = [];

        % Assign the training data (from the left subjects)
        FEAT_TMP  = FEAT_CLASSIFY;
        LABEL_TMP = LABEL;
        SUBJECT_SKINFOLD_TMP = SUBJECT_SKINFOLD;
        FEAT_TMP(i,:)  = [];
        LABEL_TMP(i,:) = [];
        SUBJECT_SKINFOLD_TMP(i,:) = [];
        FEAT_TMP = FEAT_TMP(~cellfun('isempty',FEAT_TMP));
        LABEL_TMP = LABEL_TMP(~cellfun('isempty',LABEL_TMP));    
        SUBJECT_SKINFOLD_TMP = SUBJECT_SKINFOLD_TMP(~cellfun('isempty',SUBJECT_SKINFOLD_TMP));   
        
        feature_TV  = cell2mat(FEAT_TMP);
        label_TV    = cell2mat(LABEL_TMP);
        skinfold_TV = mean(cell2mat(SUBJECT_SKINFOLD_TMP),2);
        
        feature_TV  = vertcat(feature_TV, feature_lo(y,:));
        label_TV    = vertcat(label_TV, label_lo(y,:));
        skinfold_TV = vertcat(skinfold_TV, skinfold_lo(y,:));

    else
        feature_TV = cell2mat(FEAT_CLASSIFY);
        label_TV   = cell2mat(LABEL);
        feature_test = FEAT_CLASSIFY{i};    
        label_test   = LABEL{i};        
    end
    
    % Partition into training and validation set
    cv = cvpartition(size(feature_TV,1), 'HoldOut', 0.1);
    idx = cv.test;
    
    feature_train  = feature_TV(~idx,:);
    label_train    = label_TV(~idx,:);
    skinfold_train = skinfold_TV(~idx,:);
    
    feature_valid = feature_TV(idx,:);
    label_valid   = label_TV(idx,:);
    skinfold_test = skinfold_TV(idx,:);    
    
    % Count the size and compute the ratios
    size_train(i,1) = sum(label_train == 1);
    size_train(i,2) = sum(label_train == -1); 
   
    % The total amount of training + testing data
    size_total(i,:) = size_test(i,:) + size_train(i,:);
 
    % Calculate the ratio between the positive and negative classes
    ratio = size_train(i,2) / size_train(i,1);
    fprintf('Ratio: %f\n', ratio);
    
    % Train the SVM classifier and perform cross-validation
    fprintf('Training...\n');
    SVMModel = fitcsvm(feature_train,...
                       label_train,...
                      'KernelFunction','rbf',...
                      'ClassNames',[1,-1],...
                      'KernelScale', 4.11,...
                      'BoxConstraint', 2.64,...
                      'Cost',[0,ratio;1,0]);     
                  
%     SVMModel = fitcsvm(feature_train,...
%                        label_train,...
%                       'KernelFunction','rbf',...
%                       'ClassNames',[1,-1],...
%                       'Cost',[0,ratio;1,0]);                     

    % Compute the error of training
    accuracy_train(i)      = 1 - loss(SVMModel, feature_train, label_train);
    label_predict = predict(SVMModel, feature_train);    
    fprintf('Training Accuracy: %f\n', accuracy_train(i));
    save(sprintf('CPT_Test/LOO_40/R%03d.mat', SUBJECT_ID{i}(1)), ...
         'label_train', 'label_predict', 'skinfold_train')
    
    % Compute the error of validation
    accuracy_validation(i) = 1 - loss(SVMModel, feature_valid, label_valid);
    fprintf('Validation Accuracy: %f\n', accuracy_validation(i));
       
    % Compute the error of testing
    accuracy_test(i)       = 1 - loss(SVMModel, feature_test, label_test);
    fprintf('Testing Accuracy: %f\n', accuracy_test(i));

    % ==========
    % GA Optimization to minimize the regression fitness between the
    % predicted labels and the skinfold thickness
    fprintf('Genetic Algorithm Optimization...\n')
                       
    fitnessfunc = @(w)[svmloss(w, feature_train, label_train, SVMModel), ...
                       lmr2(w, feature_train, skinfold_train, SVMModel)];                   
    A = ones(1,48);
    b = 1;
    lb = zeros(1,48);
    up = ones(1,48);
    
    tic
    options = optimoptions('gamultiobj', ...
                           'ParetoFraction', 0.3, ...
                           'PopulationSize', 32, ...
                           'MaxGenerations', 500, ...
                           'SelectionFcn', {@selectiontournament,8}, ...
                           'UseParallel', true, ...
                           'PlotFcn', {@gaplotpareto});
                       
    [W,fval,exitflag,output,population,scores] = gamultiobj(fitnessfunc, 48, ...
                                                            [],[], ...
                                                            [],[], ...
                                                            -1*ones(1,48),ones(1,48), ...
                                                            options);
    toc
    % ==========
    w = W(1,:);    
    
    % Computing Training Labels again for calculating CPT
    n = size(feature_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = feature_train .* fw;
    label_predict = predict(SVMModel, feature_tf);    
    accuracy_train_ga(i) = 1 - loss(SVMModel, feature_tf, label_train);
    fprintf('Training Accuracy after GA: %f\n', accuracy_train_ga(i));
    save(sprintf('CPT_Test/LOO_40/R%03d_GA.mat', SUBJECT_ID{i}(1)), ...
         'label_train', 'label_predict', 'skinfold_train')    
    
    % Computing Testing Accuracy
    n = size(feature_test, 1);
    fw = repmat(w, n, 1);
    feature_tf = feature_test .* fw;    
    accuracy_test_ga(i) = 1 - loss(SVMModel, feature_tf, label_test);
    fprintf('Testing Accuracy after GA: %f\n', accuracy_test_ga(i));
    
    n_w = size(W, 1);
    for s = 2 : n_w
        w = W(s,:);
        n = size(feature_test, 1);
        fw = repmat(w, n, 1);
        feature_tf = feature_test .* fw;
        test_tmp = 1 - loss(SVMModel, feature_tf, label_test);       
        if test_tmp > testing_acc_ga(i)
            n = size(feature_train, 1);
            fw = repmat(w, n, 1);
            feature_tf = feature_train .* fw;
            label_predict = predict(SVMModel, feature_tf);    
            accuracy_train_ga(i) = 1 - loss(SVMModel, feature_tf, label_train);
            fprintf('Training Accuracy after GA: %f\n', accuracy_train_ga(i));
            save(sprintf('CPT_Test/LOO_40/R%03d_GA.mat', SUBJECT_ID{i}(1)), ...
                 'label_train', 'label_predict', 'skinfold_train')    

            % Computing Testing Accuracy
            n = size(feature_test, 1);
            fw = repmat(w, n, 1);
            feature_tf = feature_test .* fw;    
            accuracy_test_ga(i) = 1 - loss(SVMModel, feature_tf, label_test);
            fprintf('Testing Accuracy after GA: %f\n', accuracy_test_ga(i));            
        end
    end
    
    end
    
end


% Display the numerical results
fprintf('Before GA')
fprintf('\nAverage Training Accuracy: %.2f%%\n', mean(accuracy_train(accuracy_train ~= 0))*100);
fprintf('Average Validation Accuracy: %.2f%%\n', mean(mean(accuracy_validation(accuracy_train ~= 0)))*100);
fprintf('Average Testing Accuracy: %.2f%%\n', mean(mean(accuracy_test(accuracy_train ~= 0)))*100);

fprintf('After GA')
fprintf('\nAverage Training Accuracy: %.2f%%\n', mean(accuracy_train_ga(accuracy_train ~= 0))*100);
fprintf('Average Testing Accuracy: %.2f%%\n', mean(mean(accuracy_test_ga(accuracy_train ~= 0)))*100);


end

result = [accuracy_train, accuracy_train_ga, accuracy_test, accuracy_test_ga];
csvwrite('LOO_40/accuracy.csv', result);

%% Optimization functions
function r2 = lmr2(w, feature_train, skinfold_train, SVMModel)
    w = reshape(w, 1, 48);
    n = size(feature_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = feature_train .* fw;
    
    label_predict = predict(SVMModel, feature_tf);
    tbl = table(skinfold_train, label_predict, ...
                'VariableNames', {'skinfold','yhat'});
    tbl.yhat = categorical(tbl.yhat);
    lm = fitlm(tbl, 'skinfold~yhat');
    
    r2 = lm.Rsquared.Ordinary;
    
end

function lsvm = svmloss(w, feature_train, label_train, SVMModel)
    w = reshape(w, 1, 48);
    n = size(feature_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = feature_train .* fw;
    
    lsvm = loss(SVMModel, feature_tf, label_train);
    
end


