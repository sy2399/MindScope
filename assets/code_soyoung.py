#!/usr/bin/env python
# coding: utf-8

# **MindScope**
#
# - Step1
#     1. Store data of each user
# - Normalize each user's data after Step1
# - Step2
#     1. Store new data (previous 4 hours)
#     2. Normalize new data
#     3. Use normalized data as input to model
#     4. Get output of model
#     5. Compare model's output and user's evaluation
#     6. Interpret models' prediction with SHAP
#
#

# TEST

# In[1]:


import pandas as pd
import os
import re

from collections import Counter
# Normalize
from sklearn.preprocessing import MinMaxScaler

# Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

import pickle
import sqlite3

# Model Interpretation
import shap

from IPython.display import Image


def filelist(dirname):
    f_list = []
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            name = os.path.join(dirname, filename)
            f_list.append(name)
    except:
        print("none")
    return f_list


files = filelist('../../Small Study/Feature_Extracted/0303')

df = pd.DataFrame()
uids = []
for f in files:
    if '.DS' not in f:
        tmp = pd.read_csv(f)
        name = re.findall('\(([^)]+)', f)[0]
        uids.append(name)
        print(name)
        #         tmp['uid'] = str(name)

        df = pd.concat([df, tmp])

# In[2]:


print(df.shape)
df.head()

# - makeLabel()
#     - 스트레스 레벨 구간화를 위한 PSS Score mean, std 계산
# - getFeatures()
#     - get John's result
# - mapLabel()
#     - Stress Score to Stress lvl
# - preprocessing
#     - replace NAN
# - normalizing
#     - MinMaxNormalize
# - initModel
#     - 1st training

# In[3]:


Image("ERD.png")


# In[4]:


# Object to save model's result
class ModelResult:

    def __init__(self, uid, dayNo, emaNo, output, accuracy, model_tag, user_tag, feature_list):
        self.uid = uid
        self.dayNo = dayNo
        self.emaNo = emaNo
        self.output = output
        self.accuracy = accuracy
        self.model_tag = model_tag
        self.user_tag = user_tag  # 0 == FAlse (default) --> 이후 유저에게 response 받으면 바뀜
        self.feature_list = feature_list

    def toString(self):
        print("\n**********", self.uid, self.dayNo, self.emaNo)
        print("Meaningful Features : ", self.feature_list)


# Object to save Features
class Feature:
    def __init__(self, category, feature_name, statement):
        self.category = category
        self.feature_name = feature_name
        self.statement = statement


class DBConnection:

    def __init__(self):
        self._createDB()

    def _createDB(self):
        # DB Connection
        db_name = "stressmodel.db"
        con = sqlite3.connect(db_name)
        cursor = con.cursor()

        create_MODEL_sql = '''Create Table if not exists ModelResult(
        Model_key integer primary key autoincrement,
        UID varchar(20) NOT NULL,
        DayNo integer NOT NULL,
        EmaNo integer NOT NULL,
        Pred_result varchar(20) NOT NULL,
        Accuracy real NOT NULL,
        Model_tag integer NOT NULL,
        User_tag integer NOT NULL)'''

        create_FEATURE_sql = '''Create Table if not exists Features(
        Feature_Key integer primary key autoincrement,
        Model_key integer NOT NULL, 
        Category text,
        Feature_name text,
        Statement text, 
        CONSTRAINT Model_key_fk FOREIGN KEY(Model_key) 
        REFERENCES ModelResult(Model_key))'''

        cursor.execute(create_MODEL_sql)
        cursor.execute(create_FEATURE_sql)

    def _insertData(self, ModelResult):
        print("\n insertDATA.....")

        # DB Connection
        db_name = "stressmodel.db"
        con = sqlite3.connect(db_name)
        cursor = con.cursor()

        insert_sql = '''insert into ModelResult(UID, DayNo, EmaNo,
        Pred_result, Accuracy, Model_tag, User_tag)
        VALUES (?, ?, ?, ?, ?, ?, ?)        
        '''
        cursor.execute(insert_sql, (ModelResult.uid, ModelResult.dayNo, ModelResult.emaNo,
                                    ModelResult.output, ModelResult.accuracy,
                                    ModelResult.model_tag, ModelResult.user_tag))
        con.commit()  # insert 문은 DML 이므로 commit 해줘야 함

        # 방금 넣은 ModelREsult 의 키값 알아야 해
        model_key = cursor.lastrowid

        for f in ModelResult.feature_list:
            print("Feature", f.category, f.feature_name, f.statement)

            insert_features_sql = '''insert into Features(Model_key,
            Category, Feature_name, Statement)
            VALUES ( ?, ?, ?, ?) '''

            cursor.execute(insert_features_sql, (model_key, f.category, f.feature_name, f.statement))
            con.commit()
        cursor.close()

    def _update(self, uid, dayNo, emaNo, user_response):
        print("\n update DB.....")

        # DB Connection
        db_name = "stressmodel.db"
        con = sqlite3.connect(db_name)
        cursor = con.cursor()

        update_sql = '''Update ModelResult set User_tag = 1
        where UID = ? and DayNo = ? and EmaNo = ? and Pred_result = ?

        '''
        cursor.execute(update_sql, (uid, dayNo, emaNo, user_response))
        con.commit()

    def _printData(self):
        print("\n printDATA....")
        # DB Connection
        db_name = "stressmodel.db"
        con = sqlite3.connect(db_name)
        cursor = con.cursor()

        cursor.execute("SELECT * FROM ModelResult")
        model_rows = cursor.fetchall()

        cursor.execute("SELECT * FROM Features")
        feature_rows = cursor.fetchall()

        return [model_rows, feature_rows]


# In[18]:


class StressModel:
    # variable for setting label
    stress_lv0_max = 0
    stress_lv1_max = 0
    stress_lv2_min = 0

    # variable for label
    CONST_STRESS_LOW = 0
    COSNT_STRESS_LITTLE_HIGH = 1
    CONST_STRESS_HIGH = 2

    feature_df_with_state = pd.read_csv('../../Feature List/feature_with_state.csv')

    def __init__(self, uid, dayNo, emaNo):

        self.uid = uid
        self.dayNo = dayNo
        self.emaNo = emaNo

    #############################################################################################

    ## CASE 1 : Right after the end of step1
    def makeLabel(self, all_df):
        """
            Step1 끝난 시점에 한번만 계산 --> 스트레스 레벨을 세 구간으로 나눔
                - just call once after step1 (for calculating stress section)
        """

        pss_mean = all_df['Stress lvl'].mean()
        pss_std = all_df['Stress lvl'].std()

        StressModel.stress_lv0_max = pss_mean
        StressModel.stress_lv1_max = pss_mean + 1.5 * pss_std
        StressModel.stress_lv2_min = pss_mean + 2 * pss_std

        print("\n makeLabel")
        print("lv0 max : ", StressModel.stress_lv0_max)
        print("lvl max : ", StressModel.stress_lv1_max)

    # -----LV0 (LOW) ----- [lv0 max] ------ LV1 (LITTLE HIGH) -----[lv1 max ]-----LV2 (HIGH)

    #############################################################################################

    def mapLabel(score):
        if score <= StressModel.stress_lv0_max:
            return StressModel.CONST_STRESS_LOW

        elif (score > StressModel.stress_lv0_max) and (score < StressModel.stress_lv1_max):
            return StressModel.CONST_STRESS_LITTLE_HIGH

        elif (score >= StressModel.stress_lv1_max):
            return StressModel.CONST_STRESS_HIGH

    def preprocessing(self, case, df):
        """
         - 1. del NAN or replace to zero
         - 2. mapping label (Stress lvl 0, 1, 2)

        """
        print(".....preprocessing")

        delNan_col = ['Audio min.', 'Audio max.', 'Audio mean', 'Sleep dur.']

        for col in df.columns:
            if (col != 'Stress lvl') & (col != 'User id') & (col != 'Day'):
                if col in delNan_col:
                    df = df[df[col] != '-']
                else:
                    df[col] = df[col].replace('-', 0)

                df[col] = pd.to_numeric(df[col])
            df = df.fillna(0)

        df['Stress_label'] = df['Stress lvl'].apply(lambda score: StressModel.mapLabel(score))

        ## Save Step1's preprocessed data in server (path ??)
        ## to use for normalize new data at step2
        if case == "step1Done":
            with open('data_result/' + str(self.uid) + "_features.p", 'wb') as file:
                pickle.dump(df, file)

        return df

    def normalizing(self, norm_type, preprocessed_df, new_row_preprocessed):
        print(".......normalizing")

        # user info columns
        userinfo = ['User id', 'Day', 'EMA order', 'Stress lvl', 'Stress_label']

        feature_scaled = pd.DataFrame()
        scaler = MinMaxScaler()

        feature_df = preprocessed_df[StressModel.feature_df_with_state['features'].values]

        if norm_type == "default":
            # feature list
            feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

        elif norm_type == "new":

            feature_df = pd.concat([feature_df.reset_index(drop=True), new_row_preprocessed[StressModel.feature_df_with_state['features'].values].reset_index(drop=True)])

            feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

        feature_scaled = pd.concat([preprocessed_df[userinfo].reset_index(drop=True), feature_scaled.reset_index(drop=True)], axis=1)

        return feature_scaled

    def initModel(self, norm_df):
        """
        initModel
        """
        print(".........initModel")

        print("===================Class Count :", Counter(norm_df['Stress_label']))
        # *** 만약 훈련 데이터에 0,1,2 라벨 중 하나가 없다면? 하나의 라벨만 존재한다면?

        X = norm_df[StressModel.feature_df_with_state['features'].values]
        Y = norm_df['Stress_label'].values

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

        model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=100)

        kfold = KFold(n_splits=10)
        scoring = {'accuracy': 'accuracy',
                   'f1_micro': 'f1_micro',
                   'f1_macro': 'f1_macro'}

        cv_results = cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring)
        model_result = {'accuracy': cv_results['test_accuracy'].mean(),
                        'f1_micro': cv_results['test_f1_micro'].mean(),
                        'f1_macro': cv_results['test_f1_macro'].mean()}

        print("===================Model Result : ", model_result)

        model.fit(X_train, Y_train)
        ## Model SAVE Path--> where?
        with open('model_result/' + str(self.uid) + "_model.p", 'wb') as file:
            pickle.dump(model, file)

    def getSHAP(self, user_all_label, pred, new_row_norm, initModel):

        shap.initjs()
        explainer = shap.TreeExplainer(initModel)
        explainer.feature_perturbation = "tree_path_dependent"

        features = StressModel.feature_df_with_state['features'].values
        feature_state_df = StressModel.feature_df_with_state

        shap_values = explainer.shap_values(new_row_norm[features])
        expected_value = explainer.expected_value

        for label in user_all_label:  # 유저한테 있는 Stress label 에 따라

            feature_list = []

            index = user_all_label.index(label)
            shap_accuracy = expected_value[index]
            shap_list = shap_values[index]

            for s_value, feature_name in zip(shap_list[0], features):
                if s_value > 0:

                    cat = feature_state_df[feature_state_df['features'] == feature_name]['category'].values[0]
                    feature_value = new_row_norm[feature_name].values[0]

                    if feature_value >= 0.5:
                        state = feature_state_df[feature_state_df['features'] == feature_name]['statement_high'].values[0]
                    else:
                        state = feature_state_df[feature_state_df['features'] == feature_name]['statement_low'].values[0]

                    #                     feature_states[cat].append(state)
                    f = Feature(cat, feature_name, state)
                    feature_list.append(f)

            print(feature_list)
            if label == pred:
                MR = ModelResult(self.uid, self.dayNo, self.emaNo, label, shap_accuracy, 1, 0, feature_list)
            else:
                MR = ModelResult(self.uid, self.dayNo, self.emaNo, label, shap_accuracy, 0, 0, feature_list)

            dbconn = DBConnection()
            dbconn._insertData(MR)
            model_tmp_df = pd.DataFrame(dbconn._printData())[0]
            feature_tmp_df = pd.DataFrame(dbconn._printData())[1]

    def update(self, user_response):
        # update Dataframe
        with open('data_result/' + str(uid) + "_features.p", 'rb') as file:
            preprocessed = pickle.load(file)
            preprocessed[(preprocessed['Day'] == self.dayNo) & (preprocessed['EMA order'] == self.emaNo)]['Stress_label'] = user_response

            with open('data_result/' + str(self.uid) + "_features.p", 'wb') as file:
                pickle.dump(preprocessed, file)

        # update Model (Retrain)
        norm_df = StressModel.normalizing(self, "default", preprocessed, None)
        StressModel.initModel(self, norm_df)

        # update DB
        dbconn = DBConnection()
        dbconn._update(self.uid, self.dayNo, self.emaNo, user_response)


#############################################################################################


# In[ ]:


def callModel(uid, case, dayNo, emaNo, user_response):
    sm = StressModel(uid, dayNo, emaNo)

    if case == "step1":
        print("********During step1....********")

    elif case == "step1Done":
        print("********Right after the end of step1...********")

        # ************ get John's result ******************
        # tc.getFeatures(uid)
        df_features = df[df['User id'] == uid]

        # preprocessing
        df_preprocessed = sm.preprocessing(case, df_features)

        # normalizing
        norm_df = sm.normalizing("default", df_preprocessed, None)

        # init model
        sm.initModel(norm_df)

    elif case == "step2":
        print("********During step2....********")
        ####################################################################################
        # get step1's preprocessed data
        with open('data_result/' + str(uid) + "_features.p", 'rb') as file:
            step1_preprocessed = pickle.load(file)

        # preprocessing new row : "without Stress lvl"
        # ************ get John's result ******************
        # new_row = getNewRow()
        df_features = df[df['User id'] == uid]
        new_row = df_features.iloc[10, :]

        new_row_df = pd.DataFrame(new_row).transpose()
        new_row_preprocessed = sm.preprocessing(case, new_row_df)

        # normalize df with new row
        norm_df = sm.normalizing("new", step1_preprocessed, new_row_preprocessed)

        ####################################################################################
        # Get Trained Model & TEST

        ## get test dasta
        new_row_for_test = norm_df[(norm_df['Day'] == dayNo) & (norm_df['EMA order'] == emaNo)]

        ## get trained model
        with open('model_result/' + str(uid) + "_model.p", 'rb') as file:
            initModel = pickle.load(file)

        features = StressModel.feature_df_with_state['features'].values

        ## test with new row
        y_pred = initModel.predict(new_row_for_test[features])

        new_row['Sterss_label'] = y_pred
        ####################################################################################

        # get SHAP result and save  (send to DB)
        user_all_labels = list(set(step1_preprocessed['Stress_label']))
        sm.getSHAP(user_all_labels, y_pred, new_row_for_test, initModel)

        # DATA- , Model UPDATE
        update_df = pd.concat([step1_preprocessed.reset_index(drop=True), new_row_preprocessed.reset_index(drop=True)])

        with open('data_result/' + str(uid) + "_features.p", 'wb') as file:
            pickle.dump(update_df, file)

    ####################################################################################

    elif case == "step2Update":

        # Model UPDATE, Data Update *****************

        # Android 로부터 유저가 선택한 값 받아와야 함.
        # new row 를 정답 라벨과 함께 새로운 데이터에 추가
        # new row 포함한 data 로 retrain 시켜서 모델 update

        sm.update(user_response)


# # After the end of Step1

# In[11]:


## uid, case, dayNo, emaNo, user_response ==> Android 로부터 받아와야 할 파라미터
uid = "hyunjae7.kim"
case = "step1Done"
dayNo = 14
emaNo = 4

user_response = None  # DATA From Android (case 가 step2Update 인 경우에만 존재)

callModel(uid, case, dayNo, emaNo, user_response)

# # During Step2

# In[12]:


## uid, case, dayNo, emaNo, user_response ==> Android 로부터 받아와야 할 파라미터
uid = "hyunjae7.kim"
case = "step2"
dayNo = 18
emaNo = 4

user_response = None  # DATA From Android (case 가 step2Update 인 경우에만 존재)

callModel(uid, case, dayNo, emaNo, user_response)

# ## Step2Update : When user response (related to output of model)

# In[19]:


## uid, case, dayNo, emaNo, user_response ==> Android 로부터 받아와야 할 파라미터
uid = "hyunjae7.kim"
case = "step2Update"
dayNo = 18
emaNo = 4

user_response = 2  # DATA From Android (case 가 step2Update 인 경우에만 존재)

callModel(uid, case, dayNo, emaNo, user_response)

# In[ ]:


# tc.preprocessing() 결과
df_preprocessed.head()

# In[ ]:


# tc.normalizing() 결과
norm_df.head()

# In[ ]:
