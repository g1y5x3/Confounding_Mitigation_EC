%% Inter-Subject Leave-One-Out Classification on Vowels
% Vocally healthy subjects vs. Vocally fatigued subjects
% 1. Assign the class based on their VFI dynamically
% 2. Subject can also be provided as an special case
% 3.  This approach assumes there are totally N subjects, the classifier 
%     is trained with N-1 subjects' data and test on the one subject left 
%     to compute the diagonsis performance.
clear
clc 

%% Load data from pre-computed features
load('data/subjects_40_v6.mat')
num_sub = 40;
num_cls = 2;
 
FEAT_CLASSIFY = FEAT_N;

%% SVM Main Loop
% Initialize a table to store the dataset spreadsheet information                
        
for lo_pen = 1 : 0.1 : 1
    size_train          = zeros(num_sub, num_cls);
    size_test           = zeros(num_sub, num_cls);
    size_total          = zeros(num_sub, num_cls);
    accuracy_train      = zeros(num_sub, 1);
    accuracy_test       = zeros(num_sub, 1);
    accuracy_validation = zeros(num_sub, 1);

    fprintf('\nLeft Out percentage: %.2f\n', lo_pen);

    % For VFI correlation analysis
    positive_per        = zeros(num_sub,1);
    VFI1_test           = zeros(num_sub,1);

    for i = 1 : 40

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
        
        fprintf('# of Healthy Samples %d\n', sum(label_TV == -1));
        fprintf('# of Fatigued Samples %d\n', sum(label_TV == 1))        

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

        fprintf('Left-out Test Samples %d\n', size(feature_test, 1));

        % Train the SVM classifier and perform cross-validation
%         SVMModel = fitcsvm(feature_train,...
%                            label_train,...
%                           'KernelFunction','rbf',...
%                           'ClassNames',[1,-1],...
%                           'KernelScale', 4.11,...
%                           'BoxConstraint', 2.64,...
%                           'Cost',[0,ratio;1,0]);
        SVMModel = fitcsvm(feature_train,...
                           label_train,...
                          'KernelFunction','rbf',...
                          'ClassNames',[1,-1],...
                          'Cost',[0,ratio;1,0]);                             
                      

        % Compute the error of training
        accuracy_train(i)      = 1 - loss(SVMModel, feature_train, label_train);
        label_predict = predict(SVMModel, feature_train);    
        fprintf('Training Accuracy: %f\n', accuracy_train(i));

        % Save the training labels, predicted labels, and skinfold thickness 
        % for calculating the p-value from CPT test.
%         save(sprintf('CPT_Test/LOO_40/R%03d.mat', SUBJECT_ID{i}(1)), ...
%              'label_train', 'label_predict', 'skinfold_train')

        % Compute the error of validation
        accuracy_validation(i) = 1 - loss(SVMModel, feature_valid, label_valid);
        fprintf('Validation Accuracy: %f\n', accuracy_validation(i));

        % Compute the error of testing
        accuracy_test(i)       = 1 - loss(SVMModel, feature_test, label_test);
        fprintf('Testing Accuracy: %f\n', accuracy_test(i));

    end

    % Display the numerical results
    fprintf('\nAverage Training Accuracy: %.2f%%\n', mean(accuracy_train(accuracy_train ~= 0))*100);
    fprintf('Average Validation Accuracy: %.2f%%\n', mean(mean(accuracy_validation(accuracy_train ~= 0)))*100);
    fprintf('Average Testing Accuracy: %.2f%%\n', mean(mean(accuracy_test(accuracy_train ~= 0)))*100);
    fprintf('Sensitivity(True Positive): %.4f\n',mean(accuracy_test(1:20)));
    fprintf('Specificity(True Negative): %.4f\n',mean(accuracy_test(21:40)));

end

%% Generate the model outputs and skinfold thickness values for CPT analysis
% feature_train = cell2mat(FEAT_CLASSIFY);
% label_train   = cell2mat(LABEL);
% 
% SVMModel = fitcsvm(feature_train,...
%                    label_train,...
%                   'KernelFunction','rbf',...
%                   'ClassNames',[1,-1],...
%                   'KernelScale', 4.11,...
%                   'BoxConstraint', 2.64);              
%               
% label_predict = predict(SVMModel, feature_train);
% 
% skinfold = cell2mat(SUBJECT_SKINFOLD);
% 
% save('CPT_Test/results_SVM_rbf.mat', ...
%      'label_train', 'label_predict', 'skinfold', ...
%      'accuracy_train', 'accuracy_test', ...
%      'SVMModel')

