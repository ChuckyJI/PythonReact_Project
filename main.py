from datetime import datetime

import pandas as pd
import numpy as np
import json
import os
import tensorflow as tf
import joblib
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def dataset1_preprocessing(file_path, paramList, test_size):
    data = pd.read_csv(file_path)
    X = data[paramList]
    Y = data.condition
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def dataset2_preprocessing(file_path, paramList, test_size):
    df = pd.read_csv(file_path)
    df.loc[df['TenYearCHD'] == 1, 'education'] = df.loc[df['TenYearCHD'] == 1, 'education'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'education'].mode()[0])
    df.loc[df['TenYearCHD'] == 0, 'education'] = df.loc[df['TenYearCHD'] == 0, 'education'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'education'].mode()[0])
    df.loc[df['TenYearCHD'] == 1, 'cigsPerDay'] = df.loc[df['TenYearCHD'] == 1, 'cigsPerDay'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'cigsPerDay'].mean())
    df.loc[df['TenYearCHD'] == 0, 'cigsPerDay'] = df.loc[df['TenYearCHD'] == 0, 'cigsPerDay'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'cigsPerDay'].mean())
    df.loc[df['TenYearCHD'] == 1, 'BPMeds'] = df.loc[df['TenYearCHD'] == 1, 'BPMeds'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'BPMeds'].mode()[0])
    df.loc[df['TenYearCHD'] == 0, 'BPMeds'] = df.loc[df['TenYearCHD'] == 0, 'BPMeds'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'BPMeds'].mode()[0])
    df.loc[df['TenYearCHD'] == 1, 'totChol'] = df.loc[df['TenYearCHD'] == 1, 'totChol'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'totChol'].mean())
    df.loc[df['TenYearCHD'] == 0, 'totChol'] = df.loc[df['TenYearCHD'] == 0, 'totChol'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'totChol'].mean())
    df.loc[df['TenYearCHD'] == 1, 'BMI'] = df.loc[df['TenYearCHD'] == 1, 'BMI'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'BMI'].mean())
    df.loc[df['TenYearCHD'] == 0, 'BMI'] = df.loc[df['TenYearCHD'] == 0, 'BMI'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'BMI'].mean())
    df.loc[df['TenYearCHD'] == 1, 'heartRate'] = df.loc[df['TenYearCHD'] == 1, 'heartRate'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'heartRate'].mean())
    df.loc[df['TenYearCHD'] == 0, 'heartRate'] = df.loc[df['TenYearCHD'] == 0, 'heartRate'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'heartRate'].mean())
    df.loc[df['TenYearCHD'] == 1, 'glucose'] = df.loc[df['TenYearCHD'] == 1, 'glucose'].fillna(
        df.loc[df['TenYearCHD'] == 1, 'glucose'].mean())
    df.loc[df['TenYearCHD'] == 0, 'glucose'] = df.loc[df['TenYearCHD'] == 0, 'glucose'].fillna(
        df.loc[df['TenYearCHD'] == 0, 'glucose'].mean())
    df.rename(columns={'totChol': 'chol', 'sysBP': 'trestbps', 'heartRate': 'thalach'}, inplace=True)
    X = df[paramList]
    Y = df.TenYearCHD
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def dataset3_preprocessing(file_path, paramList, test_size):
    df = pd.read_csv(file_path, sep=';')
    df.age = round(df.age / 365, 0)
    df.loc[df['gluc'] == 1, 'gluc'] = 0
    df.loc[df['gluc'] == 2, 'gluc'] = 1
    df.loc[df['gluc'] == 3, 'gluc'] = 1
    df.loc[df['gender'] == 2, 'gender'] = 1
    df.loc[df['gender'] == 1, 'gender'] = 0
    df.rename(columns={'gender': 'sex', 'ap_hi': 'trestbps', 'ap_lo': 'diaBP', 'gluc': 'fbs', 'smoke': 'currentSmoker'},
              inplace=True)
    X = df[paramList]
    Y = df.cardio
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def dataset4_preprocessing(file_path, paramList, test_size):
    df4 = pd.read_csv(file_path)
    df4.Smoking = df4.Smoking.map({'No': 0, 'Yes': 1})
    df4.HeartDisease = df4.HeartDisease.map({'No': 0, 'Yes': 1})
    df4.AlcoholDrinking = df4.AlcoholDrinking.map({'No': 0, 'Yes': 1})
    df4.Stroke = df4.Stroke.map({'No': 0, 'Yes': 1})
    df4.DiffWalking = df4.DiffWalking.map({'No': 0, 'Yes': 1})
    df4.Diabetic = df4.Diabetic.map({'No': 0, 'Yes': 1})
    df4.loc[df4['HeartDisease'] == 1, 'Diabetic'] = df4.loc[df4['HeartDisease'] == 1, 'Diabetic'].fillna(
        df4.loc[df4['HeartDisease'] == 1, 'Diabetic'].mode()[0])
    df4.loc[df4['HeartDisease'] == 0, 'Diabetic'] = df4.loc[df4['HeartDisease'] == 0, 'Diabetic'].fillna(
        df4.loc[df4['HeartDisease'] == 0, 'Diabetic'].mode()[0])
    df4.PhysicalActivity = df4.PhysicalActivity.map({'No': 0, 'Yes': 1})
    df4.Asthma = df4.Asthma.map({'No': 0, 'Yes': 1})
    df4.KidneyDisease = df4.KidneyDisease.map({'No': 0, 'Yes': 1})
    df4.SkinCancer = df4.SkinCancer.map({'No': 0, 'Yes': 1})
    df4.Sex = df4.Sex.map({'Female': 0, 'Male': 1})
    df4.AgeCategory = df4.AgeCategory.map(
        {'55-59': 57, '80 or older': 82, '65-69': 67, '75-79': 77, '40-44': 42, '70-74': 72, '60-64': 62, '50-54': 52,
         '45-49': 47, '18-24': 21, '35-39': 37, '30-34': 32, '25-29': 27})
    df4.Race = df4.Race.map(
        {'White': 1, 'Black': 2, 'Asian': 3, 'American Indian/Alaskan Native': 4, 'Other': 5, 'Hispanic': 6})
    df4.GenHealth = df4.GenHealth.map({'Very good': 4, 'Fair': 2, 'Good': 3, 'Poor': 1, 'Excellent': 5})
    df4.rename(
        columns={'Smoking': 'currentSmoker', 'AlcoholDrinking': 'alco', 'Stroke': 'prevalentStroke', 'Sex': 'sex',
                 'AgeCategory': 'age', 'Diabetic': 'diabetes', 'PhysicalActivity': 'active'}, inplace=True)
    X = df4[paramList]
    Y = df4.HeartDisease
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def dataset5_preprocessing(file_path, paramList, test_size):
    df5 = pd.read_csv(file_path)
    df5.General_Health = df5.General_Health.map({'Very Good': 4, 'Fair': 2, 'Good': 3, 'Poor': 1, 'Excellent': 5})
    df5.Checkup = df5.Checkup.map({'Within the past 2 years': 2, 'Within the past year': 1, '5 or more years ago': 4,
                                   'Within the past 5 years': 3, 'Never': 0})
    df5.Exercise = df5.Exercise.map({'No': 0, 'Yes': 1})
    df5.Heart_Disease = df5.Heart_Disease.map({'No': 0, 'Yes': 1})
    df5.Skin_Cancer = df5.Skin_Cancer.map({'No': 0, 'Yes': 1})
    df5.Other_Cancer = df5.Other_Cancer.map({'No': 0, 'Yes': 1})
    df5.Depression = df5.Depression.map({'No': 0, 'Yes': 1})
    df5.Arthritis = df5.Arthritis.map({'No': 0, 'Yes': 1})
    df5.Diabetes = df5.Diabetes.map({'No': 0, 'Yes': 1, 'Yes, but female told only during pregnancy': 1,
                                     'No, pre-diabetes or borderline diabetes': 0})
    df5.Smoking_History = df5.Smoking_History.map({'No': 0, 'Yes': 1})
    df5.Sex = df5.Sex.map({'Female': 0, 'Male': 1})
    df5.Age_Category = df5.Age_Category.map(
        {'55-59': 57, '80+': 82, '65-69': 67, '75-79': 77, '40-44': 42, '70-74': 72, '60-64': 62, '50-54': 52,
         '45-49': 47, '18-24': 21, '35-39': 37, '30-34': 32, '25-29': 27})
    df5.rename(columns={'Exercise': 'active', 'Skin_Cancer': 'SkinCancer', 'Diabetes': 'diabetes', 'Sex': 'sex',
                        'Age_Category': 'age', 'Height_(cm)': 'height', 'Weight_(kg)': 'weight',
                        'Smoking_History': 'currentSmoker'}, inplace=True)
    X = df5[paramList]
    Y = df5.Heart_Disease
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def model_fit_forall(x, y, cv):
    params = []
    # RandonForest Classification
    param_search_RF = [{'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 4, 8, 12, 16]},
                       {'bootstrap': [False], 'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 4, 8, 12, 16]}]
    model_RF = RandomForestClassifier(n_jobs=-1)
    grid_search_RF = GridSearchCV(model_RF, param_search_RF, cv=cv, n_jobs=-1)
    grid_search_RF.fit(x, y)
    params.append(grid_search_RF.best_params_)
    # XGboost Classification
    param_search_XG = {'learning_rate': [0.05, 0.1, 0.2, 0.3], 'n_estimators': [50, 100, 150]}
    model_XG = XGBClassifier(**{'tree_method': 'gpu_hist', 'gpu_id': 0})
    grid_search_XG = GridSearchCV(model_XG, param_search_XG, cv=cv, n_jobs=-1)
    grid_search_XG.fit(x, y)
    params.append(grid_search_XG.best_params_)
    # LGBMClassifier Classification
    param_search_LG = {'learning_rate': [0.05, 0.1, 0.3], 'n_estimators': [50, 100, 150]}
    # Acceleration in GPU {device='gpu',gpu_platform_id= 0, gpu_device_id= 0}
    model_LG = LGBMClassifier(n_jobs=-1)
    grid_search_LG = GridSearchCV(model_LG, param_search_LG, cv=cv, n_jobs=-1)
    grid_search_LG.fit(x, y)
    params.append(grid_search_LG.best_params_)
    # CatClassifier Classification
    param_search_CA = {'iterations': [300, 400, 500, 600, 700], 'learning_rate': [0.005, 0.01, 0.03, 0.05],
                       'depth': [5, 6, 7, 8]}
    model_CA = CatBoostClassifier(task_type='GPU')
    grid_search_CA = GridSearchCV(model_CA, param_search_CA, cv=cv, n_jobs=-1)
    grid_search_CA.fit(x, y)
    params.append(grid_search_CA.best_params_)
    return grid_search_RF.best_estimator_, grid_search_XG.best_estimator_, grid_search_LG.best_estimator_, grid_search_CA.best_estimator_, params


def deleteName(Object):
    if 'name' in Object:
        del Object['name']
    return Object

def testPortfolio(dict):
    for value in dict.values():
        if isinstance(value, list):
            return True


def pickUpList(dict):
    outputdic = {}
    for key,value in dict.items():
        if isinstance(value,list):
            value = [ convert_to_number(ele) for ele in value]
            outputdic[key] = value
    return outputdic

def convert_to_number(s):
    try:
        int_value = int(s)
        return int_value
    except ValueError:
        try:
            float_value = float(s)
            return float_value
        except ValueError:
            return s


def find_model_forall(model_number, file_path, paramList, test_size, test, modelList):
    # import corresponding dataset
    global x_train, y_train, x_test, y_test, logisticRegression, svc
    if model_number == 1:
        x_train, x_test, y_train, y_test = dataset1_preprocessing(file_path, paramList, test_size)
    if model_number == 2:
        x_train, x_test, y_train, y_test = dataset2_preprocessing(file_path, paramList, test_size)
    if model_number == 3:
        x_train, x_test, y_train, y_test = dataset3_preprocessing(file_path, paramList, test_size)
    if model_number == 4:
        x_train, x_test, y_train, y_test = dataset4_preprocessing(file_path, paramList, test_size)
    if model_number == 5:
        x_train, x_test, y_train, y_test = dataset5_preprocessing(file_path, paramList, test_size)

    result = []
    name = []
    params = []
    predict = []
    testdata = np.array([test])
    for ele in modelList:
        # Logistic Classification
        if ele['name'] == 'logistic':
            if testPortfolio(ele):
                param_search = pickUpList(ele)
                logisticRegression = LogisticRegression(n_jobs=-1)
                grid_search = GridSearchCV(logisticRegression, param_search, cv=3, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                params.append(grid_search.best_params_)
                logisticRegression.fit(x_train, y_train)
                name.append("Logistic")
                result.append(logisticRegression.score(x_test, y_test))
                predict.append(logisticRegression.predict(testdata))
                ele['name'] = 'logistic'
            else:
                deleteName(ele)['n_jobs'] = -1
                logisticRegression = LogisticRegression(**deleteName(ele)).fit(x_train, y_train)
                name.append("Logistic")
                params.append(deleteName(ele))
                result.append(logisticRegression.score(x_test, y_test))
                predict.append(logisticRegression.predict(testdata))
                ele['name']='logistic'

        # SVC Classification
        if ele['name'] == 'svc':
            if testPortfolio(ele):
                param_search = pickUpList(ele)
                svc = SVC()
                grid_search = GridSearchCV(svc, param_search, cv=3, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                params.append(grid_search.best_params_)
                svc.fit(x_train, y_train)
                name.append("SVC")
                result.append(svc.score(x_test, y_test))
                predict.append(svc.predict(testdata))
                ele['name'] = 'svc'
            else:
                svc = SVC(**deleteName(ele)).fit(x_train, y_train)
                name.append("SVC")
                params.append(deleteName(ele))
                result.append(svc.score(x_test, y_test))
                predict.append(svc.predict(testdata))
                ele['name'] = 'svc'

        # Random forest
        if ele['name'] == 'randomForest':
            if testPortfolio(ele):
                param_search = pickUpList(ele)
                randomForestClassifier = RandomForestClassifier(n_jobs=-1)
                grid_search = GridSearchCV(randomForestClassifier, param_search, cv=3, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                params.append(grid_search.best_params_)
                randomForestClassifier.fit(x_train, y_train)
                name.append("Random Forest")
                result.append(randomForestClassifier.score(x_test, y_test))
                predict.append(randomForestClassifier.predict(testdata))
                ele['name'] = 'randomForest'
            else:
                deleteName(ele)['n_jobs'] = -1
                randomForestClassifier = RandomForestClassifier(**deleteName(ele)).fit(x_train, y_train)
                name.append("Random Forest")
                params.append(deleteName(ele))
                result.append(randomForestClassifier.score(x_test, y_test))
                predict.append(randomForestClassifier.predict(testdata))
                ele['name'] = 'randomForest'

        # XGBoost
        if ele['name'] == 'XGBoost':
            if testPortfolio(ele):
                param_search = pickUpList(ele)
                xGBClassifier = XGBClassifier(**{'tree_method': 'gpu_hist', 'gpu_id': 0})
                grid_search = GridSearchCV(xGBClassifier, param_search, cv=3, n_jobs=-1)
                grid_search.fit(x_train, y_train)
                params.append(grid_search.best_params_)
                xGBClassifier.fit(x_train, y_train)
                name.append("XGBoost")
                result.append(xGBClassifier.score(x_test, y_test))
                predict.append(xGBClassifier.predict(testdata))
                ele['name'] = 'XGBoost'
            else:
                deleteName(ele)['tree_method'] = 'gpu_hist'
                deleteName(ele)['gpu_id'] = 0
                xGBClassifier = XGBClassifier(**deleteName(ele)).fit(x_train, y_train)
                name.append("XGBoost")
                params.append(deleteName(ele))
                result.append(xGBClassifier.score(x_test, y_test))
                predict.append(xGBClassifier.predict(testdata))
                ele['name'] = 'XGBoost'


    # # LGBM Classification
    # model_LG.fit(x_train, y_train)
    # name.append("LightGBM")
    # result.append(model_LG.score(x_test, y_test))
    #
    # # Cat Classification
    # model_CA.fit(x_train, y_train)
    # name.append("CatBoost")
    # result.append(model_CA.score(x_test, y_test))
    #
    # # NN
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #
    # def one_hot(y_list):
    #     global _oneHot
    #     oneHot = []
    #     for y in y_list:
    #         if y == 1:
    #             _oneHot = [1, 0]
    #         if y == 0:
    #             _oneHot = [0, 1]
    #         oneHot.append(_oneHot)
    #     return np.array(oneHot)
    #
    # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    # dataset = dataset.shuffle(buffer_size=10000)
    # dataset = dataset.batch(batch_size=10000)
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #
    # model_NN = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    #     tf.keras.layers.Dense(32, activation='relu'),
    #     tf.keras.layers.Dense(16, activation='relu'),
    #     tf.keras.layers.Dense(2, activation='softmax')
    # ])
    #
    # for batch in dataset:
    #     features, labels = batch
    #     labels = one_hot(labels)
    #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1,
    #                                                       restore_best_weights=True)
    #     model_NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     model_NN.fit(features, labels, epochs=300, callbacks=[early_stopping])
    #
    # a = model_NN.evaluate(x_test, one_hot(y_test))
    # name.append("Neural Network")
    # result.append(a[1])
    #
    # # joblib save the model params
    # joblib.dump(logisClassification, "logisClassification.pkl")
    # joblib.dump(svcClassification, "svcClassification.pkl")
    # joblib.dump(model_RF, "model_RF.pkl")
    # joblib.dump(model_XG, "model_XG.pkl")
    # joblib.dump(model_LG, "model_LG.pkl")
    # joblib.dump(model_CA, "model_CA.pkl")

    return predict, result, params, name


def assignModel(json_data):
    # generate the columns and unify names
    column1 = pd.read_csv('/root/autodl-tmp/Data/dataset_1.csv').columns.tolist()
    # column1 = pd.read_csv('Data/dataset_1.csv').columns.tolist()
    column1.remove('condition')

    column2 = pd.read_csv('/root/autodl-tmp/Data/dataset_2.csv').columns.tolist()
    # column2 = pd.read_csv('Data/dataset_2.csv').columns.tolist()
    column2[10] = 'trestbps'
    column2[9] = 'chol'
    column2[13] = 'thalach'
    column2.remove('TenYearCHD')

    column3 = pd.read_csv('/root/autodl-tmp/Data/dataset_3.csv', sep=';').columns.tolist()
    # column3 = pd.read_csv('Data/dataset_3.csv', sep=';').columns.tolist()
    column3[2] = 'sex'
    column3[5] = 'trestbps'
    column3[6] = 'diaBP'
    column3[8] = 'fbs'
    column3[9] = 'currentSmoker'
    column3.remove('id')
    column3.remove('cardio')

    column4 = pd.read_csv('/root/autodl-tmp/Data/dataset_4.csv').columns.tolist()
    # column4 = pd.read_csv('Data/dataset_4.csv').columns.tolist()
    column4[2] = 'currentSmoker'
    column4[3] = 'alco'
    column4[4] = 'prevalentStroke'
    column4[8] = 'sex'
    column4[9] = 'age'
    column4[11] = 'diabetes'
    column4[12] = 'active'
    column4.remove('HeartDisease')

    column5 = pd.read_csv('/root/autodl-tmp/Data/dataset_5.csv').columns.tolist()
    # column5 = pd.read_csv('Data/dataset_5.csv').columns.tolist()
    column5[2] = 'active'
    column5[4] = 'SkinCancer'
    column5[7] = 'diabetes'
    column5[9] = 'sex'
    column5[10] = 'age'
    column5[11] = 'height'
    column5[12] = 'weight'
    column5[14] = 'currentSmoker'
    column5.remove('Heart_Disease')

    # processing json data
    json_data = json_data.replace("'", "\"")
    json_data = json.loads(json_data)
    keys = json_data.keys()
    values = json_data.values()
    keysList = list(keys)
    valuesList = list(values)
    keysList1, valuesList1 = [], []
    keysList2, valuesList2 = [], []
    keysList3, valuesList3 = [], []
    keysList4, valuesList4 = [], []
    keysList5, valuesList5 = [], []

    for ele in range(len(keysList)):
        if keysList[ele] in column1:
            keysList1.append(keysList[ele])
            valuesList1.append(valuesList[ele])
        if keysList[ele] in column2:
            keysList2.append(keysList[ele])
            valuesList2.append(valuesList[ele])
        if keysList[ele] in column3:
            keysList3.append(keysList[ele])
            valuesList3.append(valuesList[ele])
        if keysList[ele] in column4:
            keysList4.append(keysList[ele])
            valuesList4.append(valuesList[ele])
        if keysList[ele] in column5:
            keysList5.append(keysList[ele])
            valuesList5.append(valuesList[ele])

    resultRecord = [keysList1, keysList2, keysList3, keysList4, keysList5]

    valuesRecord = [valuesList1, valuesList2, valuesList3, valuesList4, valuesList5]

    maxNumber = np.argmax([len(keysList1), len(keysList2), len(keysList3), len(keysList4), len(keysList5)])

    return resultRecord[maxNumber], valuesRecord[maxNumber], maxNumber


def dataPost(jsonstring):
    jsonString, dataType, sampleID, modelist = transformRequest(jsonstring)
    paramList, numberList, number = assignModel(jsonString)
    number = number + 1
    docList = ['0', '/root/autodl-tmp/Data/dataset_1.csv', '/root/autodl-tmp/Data/dataset_2.csv',
               '/root/autodl-tmp/Data/dataset_3.csv', '/root/autodl-tmp/Data/dataset_4.csv',
               '/root/autodl-tmp/Data/dataset_5.csv']
    result, resultList, paramsList,nameList = find_model_forall(number, docList[number], paramList, 0.15, numberList,modelist)
    # id fot each patient
    outputjson = {"result": result, "resultList": resultList, "numberOfDataset": number, "paramsList": paramsList, "sampleID": sampleID, "name":nameList}
    print(outputjson)
    return outputjson

def transformRequest(originalRequest):
    jsonstring = eval(originalRequest['data'])
    if 'dataType' in jsonstring:
        del jsonstring['dataType']
    jsonstring = str(jsonstring)
    dataType = eval(originalRequest['data'])['dataType']
    sampleID = originalRequest['id']
    modelist = json.loads(originalRequest['model'])
    return jsonstring,dataType,sampleID,modelist


@app.route('/gettestdata', methods=['OPTIONS'])
def handle_options():
    response_headers = {
        'Access-Control-Allow-Origin': '*',  # Adjust this based on your CORS requirements
        'Access-Control-Allow-Methods': 'POST',  # Add more methods if needed
        'Access-Control-Allow-Headers': 'Content-Type'  # Add more headers if needed
    }
    return '', 204, response_headers


@app.route('/gettestdata', methods=['POST'])
def postdata():
    getjson = request.get_json()
    if getjson:
        result = dataPost(getjson)
        return result
    else:
        return "No JSON data received", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3010)
    # a = {'id': 'sampleIDString', 'data': '{"sex":0,"age":43,"education":2,"currentSmoker":0,"cigsPerDay":0,"BPMeds":0,"prevalentStroke":0,"prevalentHyp":0,"diabetes":0,"chol":247,"trestbps":131,"diaBP":88,"BMI":27.64,"thalach":72,"glucose":61,"KidneyDisease":1,"dataType":"b"}', 'model': '[{"name":"randomForest","n_estimators":["60","80","120"],"max_depth":["4","6"],"n_jobs":-1},{"name":"XGBoost","n_estimators":100,"max_depth":6,"learning_rate":"0.2","min_child_weight":1,"subsample":1,"colsample_bytree":1}]'}
    # dataPost(a)

