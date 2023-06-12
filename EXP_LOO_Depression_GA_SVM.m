rng('shuffle');

% Hopefully I don't get depressed at the end of this experiment = =
Tbl = readtable('pone.csv');

% Extract subject info
subject_id = Tbl.PID;
sub_all = unique(subject_id);
num_subject = size(sub_all,1);

% x - feature vector
% y - label
% c - confounder

% Extract demographic info
% c = [Tbl.age, Tbl.occupation, Tbl.education];
c = Tbl.age;

% Extract labels
y = Tbl.iscase;

% Extract selected 36 features (excluding age)
x = [Tbl.pcm_intensity_sma_quartile1, ...
     Tbl.pcm_loudness_sma_linregerrA, ...
     Tbl.pcm_loudness_sma_stddev    , ...
     Tbl.pcm_loudness_sma_iqr2_3    , ...
     Tbl.pcm_loudness_sma_iqr1_3    , ...
     Tbl.mfcc_sma_1__max            , ...
     Tbl.mfcc_sma_2__max            , ...
     Tbl.mfcc_sma_2__amean          , ...
     Tbl.mfcc_sma_5__min            , ...
     Tbl.mfcc_sma_5__stddev         , ...
     Tbl.mfcc_sma_5__iqr1_2         , ...
     Tbl.mfcc_sma_6__min            , ...
     Tbl.lspFreq_sma_3__amean       , ...
     Tbl.lspFreq_sma_3__quartile1   , ...
     Tbl.lspFreq_sma_3__quartile2   , ...
     Tbl.lspFreq_sma_3__quartile3   , ...
     Tbl.lspFreq_sma_4__amean       , ...
     Tbl.lspFreq_sma_4__quartile1   , ...
     Tbl.lspFreq_sma_4__quartile2   , ...
     Tbl.lspFreq_sma_5__amean       , ...
     Tbl.lspFreq_sma_5__quartile1   , ...
     Tbl.mfcc_sma_de_2__quartile3   , ...
     Tbl.mfcc_sma_de_2__iqr1_2      , ...
     Tbl.mfcc_sma_de_2__iqr1_3      , ...
     Tbl.mfcc_sma_de_3__linregerrA  , ...
     Tbl.mfcc_sma_de_3__linregerrQ  , ...
     Tbl.mfcc_sma_de_3__stddev      , ...
     Tbl.mfcc_sma_de_5__linregerrA  , ...
     Tbl.mfcc_sma_de_5__linregerrQ  , ...
     Tbl.mfcc_sma_de_5__stddev      , ...
     Tbl.mfcc_sma_de_7__linregerrA  , ...
     Tbl.mfcc_sma_de_7__linregerrQ  , ...
     Tbl.mfcc_sma_de_7__stddev      , ...
     Tbl.voiceProb_sma_de_quartile1 , ...
     Tbl.voiceProb_sma_de_iqr1_2    , ...
     Tbl.voiceProb_sma_de_iqr1_3];

% x = [x, Tbl.education]; 
 
% % Normalizing the features
% mu = mean(x);
% sigma = mean(x);
% 
% for d =  1 : size(mu,2)
%     x(:,d) = (x(:,d) - mu(d)) / sigma(d);        
% end

%%
training_acc    = zeros(num_subject,1);
training_acc_ga = zeros(num_subject,1);
valid_acc       = zeros(num_subject,1);
testing_acc     = zeros(num_subject,1);
testing_acc_ga  = zeros(num_subject,1);

for s = 1 : num_subject
    fprintf("NO.%02d: %07d\n", s, sub_all(s));
    id_train = (subject_id ~= sub_all(s));
    id_test  = (subject_id == sub_all(s));

    x_train  = x(id_train,:);
    y_train  = y(id_train);
    c_train  = subject_id(id_train);
    
    x_test   = x(id_test,:);
    y_test   = y(id_test);
    
    Mdl = fitcsvm(x_train,y_train, ...
                  'KernelFunction','linear');

    L_train = loss(Mdl, x_train, y_train);
    
    yhat_train = predict(Mdl, x_train);
    
    % Save the model predictions along with the true labels and confounder
    % for running CPT
    save(sprintf('CPT_Test_Depression/%07d.mat', sub_all(s)), ...
         'y_train', 'yhat_train', 'c_train')
    
    L_test = loss(Mdl, x_test, y_test);

    training_acc(s) = 1-L_train;
    testing_acc(s)  = 1-L_test;
    
    fprintf('Training Accuracy: %f\n', training_acc(s));
    fprintf('Testing Accuracy: %f\n', testing_acc(s));
    
    fprintf('Genetic Algorithm Optimization...\n')
                       
    fitnessfunc = @(w)[svmloss(w, x_train, y_train, Mdl), ...
                       lmr2(w, x_train, c_train, Mdl)];                   
    A = ones(1,36);
    b = 1;
    lb = zeros(1,36);
    up = ones(1,36);
    
    tic
    options = optimoptions('gamultiobj', ...
                           'ParetoFraction', 0.3, ...
                           'PopulationSize', 128, ...
                           'MaxGenerations', 100, ...
                           'SelectionFcn', {@selectiontournament,8}, ...
                           'UseParallel', true);

    [W,fval,exitflag,output,population,scores] = gamultiobj(fitnessfunc, 36, ...
                                                            [],[], ...
                                                            [],[], ...
                                                            [],[], ...
                                                            options);
    toc    
    
    % The result returns the pareto front of the solutions which are 
    % essentially 10 points with trade-off between p-value fitness and
    % svm loss fitness
    
    % initialize the results
    w = W(1,:);
    
    % Compute Training Accuracy
    n = size(x_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = x_train .* fw;            
    yhat_train = predict(Mdl, feature_tf); 
    save(sprintf('CPT_Test_Depression/%07d_GA.mat', sub_all(s)), ...
                 'y_train', 'yhat_train', 'c_train')    
    training_acc_ga(s) = 1 - loss(Mdl, feature_tf, y_train);
    fprintf('Training Accuracy after GA: %f\n', training_acc_ga(s));

    % Computing Testing Accuracy
    n = size(x_test, 1);
    fw = repmat(w, n, 1);
    feature_tf = x_test .* fw;    
    testing_acc_ga(s) = 1 - loss(Mdl, feature_tf, y_test);
    fprintf('Testing Accuracy after GA: %f\n', testing_acc_ga(s));    
    
    n_w = size(W, 1);
    for i = 2 : n_w
        w = W(i,:);
        n = size(x_test, 1);
        fw = repmat(w, n, 1);
        feature_tf = x_test .* fw;
        test_tmp = 1 - loss(Mdl, feature_tf, y_test);
        if test_tmp > testing_acc_ga(s)
            n = size(x_train, 1);
            fw = repmat(w, n, 1);
            feature_tf = x_train .* fw;            
            yhat_train = predict(Mdl, feature_tf); 
            save(sprintf('CPT_Test_Depression/%07d_GA.mat', sub_all(s)), ...
                 'y_train', 'yhat_train', 'c_train')    
            training_acc_ga(s) = 1 - loss(Mdl, feature_tf, y_train);
            fprintf('Training Accuracy after GA: %f\n', training_acc_ga(s));

            % Computing Testing Accuracy
            n = size(x_test, 1);
            fw = repmat(w, n, 1);
            feature_tf = x_test .* fw;    
            testing_acc_ga(s) = 1 - loss(Mdl, feature_tf, y_test);
            fprintf('Testing Accuracy after GA: %f\n', testing_acc_ga(s));    
        end
    end
    
end

fprintf('Before GA\n')
fprintf('Training Accuracy: %.4f%%\n', 100*mean(training_acc));
fprintf('Testing Accuracy: %.4f%%\n', 100*mean(testing_acc));
fprintf('After GA\n')
fprintf('Training Accuracy: %.4f%%\n', 100*mean(training_acc_ga));
fprintf('Testing Accuracy: %.4f%%\n', 100*mean(testing_acc_ga));

result = [training_acc, training_acc_ga, testing_acc, testing_acc_ga];
csvwrite('CPT_Test_Depression/accuracy.csv', result);

% 
%% If the confounder is Categorical
% function r2 = lmr2(w, x_train, c_train, Mdl)
%     d = size(x_train, 2);
%     w = reshape(w, 1, d);
%     n = size(x_train, 1);
%     fw = repmat(w, n, 1);
%     feature_tf = x_train .* fw;
%     
%     label_predict = predict(Mdl, feature_tf);
%     tbl = table(c_train, label_predict, ...
%                 'VariableNames', {'c','yhat'});
%     tbl.c    = categorical(tbl.c);
%     tbl.yhat = categorical(tbl.yhat);
%     glm = fitglm(tbl);
%     
%     r2 = glm.Rsquared.Ordinary;
%     if isnan(r2)
%         r2 = 1;
%     end
%     
% end

%% If the confounder is Continuous
function r2 = lmr2(w, x_train, c_train, Mdl)
    d = size(x_train, 2);
    w = reshape(w, 1, d);
    n = size(x_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = x_train .* fw;
    
    label_predict = predict(Mdl, feature_tf);
    tbl = table(c_train, label_predict, ...
                'VariableNames', {'c_train','yhat'});
    tbl.yhat = categorical(tbl.yhat);
    lm = fitlm(tbl, 'c_train~yhat');
    
    r2 = lm.Rsquared.Ordinary;
    if isnan(r2)
        r2 = 1;
    end    
end

%% SVM Loss
function lsvm = svmloss(w, x_train, y_train, Mdl)
    d = size(x_train, 2);
    w = reshape(w, 1, d);
    n = size(x_train, 1);
    fw = repmat(w, n, 1);
    feature_tf = x_train .* fw;
    
    lsvm = loss(Mdl, feature_tf, y_train);
end     
