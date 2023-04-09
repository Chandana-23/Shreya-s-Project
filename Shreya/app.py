import streamlit as st

import math, time, random, datetime
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os



X = pd.read_csv('Shreya/X.csv')
Y = pd.read_csv('Shreya/Y.csv')

X = X.drop(columns=['Unnamed: 0'])
Y = Y.drop(columns=['Unnamed: 0'])

Y = np.where(Y==0.0,0,1)
Y = np.array(Y)

# Define a dictionary of models and their corresponding classes
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gaussian Naive Bayes":GaussianNB(),
    "SVM":LinearSVC(),
    "Gradient Boosting Classifier":GradientBoostingClassifier()
}


def evaluate_model(_model):
    start_time = time.time()
    if _model == "Logistic Regression":
        _model = pickle.load(open('lr.sav', 'rb'))

    elif _model == "Decision Trees":
        _model = pickle.load(open('dt.sav', 'rb'))

    elif _model == "Random Forest":
        _model = pickle.load(open('rf.sav', 'rb'))

    elif _model == "Gaussian Naïve Bayes":
        _model = pickle.load(open('gnb.sav', 'rb'))

    elif _model == "SVM":
        _model = pickle.load(open('svc.sav', 'rb'))

    elif _model == "Gradient Boosting Classifier":
        _model = pickle.load(open('gbt.sav', 'rb'))
        
    _model.fit(X,Y)
    lr_time = (time.time() - start_time)
    
    y_pred = model_selection.cross_val_predict(_model, X, Y, cv=10, n_jobs = -1)

    # Compute evaluation metrics
    acc = round(_model.score(X,Y)*100,2)
    accuracy = round(accuracy_score(Y, y_pred)*100, 2)
    precision = round(precision_score(Y, y_pred, pos_label = 0),2)
    recall = round(recall_score(Y, y_pred, pos_label = 0),2)
    f1 = f1_score(Y, y_pred, average='weighted')

    return acc,accuracy,precision,recall,f1,lr_time
    
left_column, right_column = st.columns([2,3])

# Add content to the left column
with left_column:
    
    st.title("Machine Learning Model Evaluation")
    model_name = st.selectbox("Select a model", list(models.keys()))
    model = models[model_name]
    acc,accuracy,precision,recall,f1,lr_time = evaluate_model(model)
    st.write("Accuracy: ",acc)
    st.write("Accuracy of CV:", accuracy)
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1)
    st.write("Time:",lr_time)




# Add content to the right column
with right_column:
    st.title("Metrics evaluation Vs Model")
    algo_name = ['Log. Reg.','Decision Tree','RandomForest Gini','Gaussian NB','SVM','GradientBoosting']
    a = [97.58, 98.54, 98.42, 67.64, 97.72, 98.72]
    t = [3.978604793548584,
        1.7837371826171875,
        9.503568172454834,
        0.7985720634460449,
        4.859381437301636,
        25.48365592956543]
    p = [0.95, 0.88, 0.95, 0.13, 0.94, 0.95]
    r = [0.53, 0.81, 0.72, 1.0, 0.56, 0.78]
    df = pd.DataFrame(pd.DataFrame({'Algorithm' : algo_name, 'Accuracy %' : a, 'Time taken':t, 'Precision':p, 'Recall':r}))
    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.bar(df['Algorithm'], df['Accuracy %'])
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Accuracy(%)')
    st.pyplot(fig)


    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.bar(df['Algorithm'], df['Time taken'])
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Time taken')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.bar(df['Algorithm'], df['Precision'])
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Precision')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    plt.xticks(rotation=90)
    ax.bar(df['Algorithm'], df['Recall'])
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Recall')
    st.pyplot(fig)

    

