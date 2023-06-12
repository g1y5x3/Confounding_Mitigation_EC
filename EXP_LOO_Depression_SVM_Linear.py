#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score


data = pd.read_csv('data/pone.csv')

subject_id  = data['PID'].to_numpy()
subject_all = data['PID'].unique()
num_subject = len(subject_all)

x = data[['pcm_intensity_sma_quartile1', 
          'pcm_loudness_sma_linregerrA',
          'pcm_loudness_sma_stddev',
          'pcm_loudness_sma_iqr2.3',
          'pcm_loudness_sma_iqr1.3',
          'mfcc_sma.1._max',
          'mfcc_sma.2._max',
          'mfcc_sma.2._amean',
          'mfcc_sma.5._min',
          'mfcc_sma.5._stddev',
          'mfcc_sma.5._iqr1.2',
          'mfcc_sma.6._min',
          'lspFreq_sma.3._amean',
          'lspFreq_sma.3._quartile1',
          'lspFreq_sma.3._quartile2',
          'lspFreq_sma.3._quartile3',
          'lspFreq_sma.4._amean',
          'lspFreq_sma.4._quartile1',
          'lspFreq_sma.4._quartile2',
          'lspFreq_sma.5._amean',
          'lspFreq_sma.5._quartile1',
          'mfcc_sma_de.2._quartile3',
          'mfcc_sma_de.2._iqr1.2',
          'mfcc_sma_de.2._iqr1.3',
          'mfcc_sma_de.3._linregerrA',
          'mfcc_sma_de.3._linregerrQ',
          'mfcc_sma_de.3._stddev',
          'mfcc_sma_de.5._linregerrA',
          'mfcc_sma_de.5._linregerrQ',
          'mfcc_sma_de.5._stddev',
          'mfcc_sma_de.7._linregerrA',
          'mfcc_sma_de.7._linregerrQ',
          'mfcc_sma_de.7._stddev',
          'voiceProb_sma_de_quartile1',
          'voiceProb_sma_de_iqr1.2',
          'voiceProb_sma_de_iqr1.3']].to_numpy()

y = data['iscase'].to_numpy()

training_acc = np.zeros((num_subject,1))
testing_acc  = np.zeros((num_subject,1))

for s in range(num_subject):
    print('NO.{}: {}'.format(s, subject_all[s]))
    id_train = (subject_id != subject_all[s])
    id_test  = (subject_id == subject_all[s])
    
    x_train = x[id_train,:]
    y_train = y[id_train]
    
    x_test  = x[id_test,:]
    y_test  = y[id_test]
    
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    
    label_predict = clf.predict(x_train)
    
    print('Training Acc: ', accuracy_score(label_predict, y_train))
    training_acc[s] = accuracy_score(label_predict, y_train)
    
    label_predict = clf.predict(x_test)
    print('Testing Acc: ', accuracy_score(label_predict, y_test))
    testing_acc[s] = accuracy_score(label_predict, y_test)    
    
print('Average Training Accuracy: {0:.2f}%'.format(100*np.mean(training_acc)))
print('Average Testing Accuracy: {0:.2f}%'.format(100*np.mean(testing_acc)))