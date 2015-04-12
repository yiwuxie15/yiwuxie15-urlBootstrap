# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:11:47 2015

@author: Wizard
"""

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier




with open('final_dataset.json_shuffled_deduped.json') as data_file:
    data = json.load(data_file)

#count_r = 0
#count_p = 0
#for page in data:
#    if page['plabel'] == 1:
#        count_p += 1
#    if page['rlabel'] == 1:
#        count_r += 1
#
#print count_r, count_p

X_person_train = []
X_research_train = []
y_person_train = []
y_research_train = []

for page in data:
    if page['plabel'] != -1:
        X_text = page['url'].replace('/', ' ').replace('?', ' ').replace('.', ' ')
        y_plabel = page['plabel']
        y_rlabel = page['rlabel']
        X_person_train.append(X_text)
        X_research_train.append(X_text)
        y_person_train.append(y_plabel)
        y_research_train.append(y_rlabel)

c = CountVectorizer(decode_error = 'ignore')
X_person_train = c.fit_transform(X_person_train)
X_person_train = X_person_train.todense()
#X_research_train = c.fit_transform(X_research_train)


def train_classifiers(X_data, y_data):
    ############ Linear SVM: 0.908 #############
    clf_LSVM = svm.SVC(kernel = 'linear')
    clf_LSVM.fit(X_data, y_data)
    
    ############ MultinomialNB: 0.875 #############
    clf_MNB = MultinomialNB()
    clf_MNB.fit(X_data, y_data)
    
    ############ Random Forest: 0.910 #############
    clf_RF = RandomForestClassifier(n_estimators=200, criterion='entropy')
    clf_RF.fit(X_data, y_data)
    
    ############ Extra Tree: 0.915 ##################
    clf_ETC = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0)
    clf_ETC.fit(X_data, y_data)
    
    ############ AdaBoost: 0.88 ##################
    clf_Ada = AdaBoostClassifier()
    clf_Ada.fit(X_data, y_data)
    
    ############ rbf SVM: 0.895 #############
    clf_rbf = svm.SVC(C=200, gamma=0.06, kernel='rbf')
    clf_rbf.fit(X_data, y_data)
    
    ############ GradientBoosting: 0.88 #############
    clf_GBC = GradientBoostingClassifier()
    clf_GBC.fit(X_data, y_data)
    
    return clf_LSVM, clf_MNB, clf_RF, clf_ETC, clf_Ada, clf_rbf, clf_GBC    
    #scores = cross_val_score(clf_LSVM, X_person_train, y_person_train)
    #print scores.mean()
    #print y_person_train.count(1)*1.0/len(y_person_train)

clf_LSVM, clf_MNB, clf_RF, clf_ETC, clf_Ada, clf_rbf, clf_GBC = train_classifiers(X_person_train, y_person_train)

reTrain_count = 0
for page in data:
    if page['plabel'] == -1:
        X_text = [page['url'].replace('/', ' ').replace('?', ' ').replace('.', ' ')]
        X_text = c.transform(X_text)
        X_text = X_text.todense()
        if reTrain_count == 100:
            clf_LSVM, clf_MNB, clf_RF, clf_ETC, clf_Ada, clf_rbf, clf_GBC = train_classifiers(X_person_train, y_person_train)
            reTrain_count = 0
        y_LSVM = clf_LSVM.predict(X_text)
        y_MNB = clf_MNB.predict(X_text)[0]
        y_RF = clf_RF.predict(X_text)[0]
        y_ETC = clf_ETC.predict(X_text)[0]
        y_Ada = clf_Ada.predict(X_text)[0]
        y_rbf = clf_rbf.predict(X_text)[0]
        y_GBC = clf_GBC.predict(X_text)[0]
        print (y_LSVM, y_MNB, y_RF, y_ETC, y_Ada, y_rbf, y_GBC)
        if y_LSVM == y_MNB == y_RF == y_ETC == y_Ada == y_rbf == y_GBC == 1:
            page['plabel'] = 1
            X_person_train.append(X_text)
            y_person_train.append(1)
        elif y_LSVM == y_MNB == y_RF == y_ETC == y_Ada == y_rbf == y_GBC == 0:
            page['plabel'] = 0
            X_person_train.append(X_text)
            y_person_train.append(0)
            reTrain_count += 1
        print reTrain_count
        
with open('annotated_bootstraping.json', 'wb') as pages:
    json.dump(data, pages)

pages.close()
        

        
    