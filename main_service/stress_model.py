import pickle

import statistics
from collections import Counter
# Normalize
from sklearn.preprocessing import MinMaxScaler

# Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

import shap
import pandas as pd

from main_service.models import ModelResult
from main_service.models import AppUsed


## CASE 1 : Right after the end of step1
def makeLabel(all_stress_levels):
    """
        Step1 끝난 시점에 한번만 계산 --> 스트레스 레벨을 세 구간으로 나눔
            - just call once after step1 (for calculating stress section)
    """

    pss_mean = statistics.mean(all_stress_levels)
    pss_std = statistics.stdev(all_stress_levels)

    stress_lv0_max = pss_mean
    stress_lv1_max = pss_mean + 1.5 * pss_std
    stress_lv2_min = pss_mean + 2 * pss_std

    # print("\n makeLabel")
    # print("lv0 max : ", StressModel.stress_lv0_max)
    # print("lvl max : ", StressModel.stress_lv1_max)

    # StressModel.stress_lv0_max = pss_mean
    # StressModel.stress_lv1_max = pss_mean + 1.5 * pss_std
    # StressModel.stress_lv2_min = pss_mean + 2 * pss_std

    return [stress_lv0_max, stress_lv1_max, stress_lv2_min]


class StressModel:
    # variable for setting label

    # variable for label
    CONST_STRESS_LOW = 0
    CONST_STRESS_LITTLE_HIGH = 1
    CONST_STRESS_HIGH = 2

    feature_df_with_state = pd.read_csv('assets/feature_with_state.csv')

    def __init__(self, uid, dayNo, emaNo, stress_lv0_max, stress_lv1_max, stress_lv2_min):
        self.uid = uid
        self.dayNo = dayNo
        self.emaNo = emaNo
        self.stress_lv0_max = stress_lv0_max
        self.stress_lv1_max = stress_lv1_max
        self.stress_lv2_min = stress_lv2_min

    def mapLabel(self, score):
        try:
            if score <= self.stress_lv0_max:
                return StressModel.CONST_STRESS_LOW

            elif (score > self.stress_lv0_max) and (score < self.stress_lv1_max):
                return StressModel.CONST_STRESS_LITTLE_HIGH

            elif (score >= self.stress_lv1_max):
                return StressModel.CONST_STRESS_HIGH
        except Exception as e:
            print(e)

    def preprocessing(self, df, prep_type):
        """
         - 1. del NAN or replace to zero
         - 2. mapping label (Stress lvl 0, 1, 2)

        """
        print(".....preprocessing")

        delNan_col = ['Audio min.', 'Audio max.', 'Audio mean', 'Sleep dur.']
        try:
            for col in df.columns:
                if (col != 'Stress lvl') & (col != 'User id') & (col != 'Day'):
                    df[col] = df[col].replace('-', 0)

                    df[col] = pd.to_numeric(df[col])
                df = df.fillna(0)

            if prep_type == "default":
                df['Stress_label'] = df['Stress lvl'].apply(lambda score: self.mapLabel(score))
            else:
                df['Stress_label'] = -1
                df['Stress lvl'] = -1

        except Exception as e:
            print(e)

        return df

    def normalizing(self, norm_type, preprocessed_df, new_row_preprocessed, user_email, day_num, ema_order):
        print(".......normalizing")

        # user info columns
        userinfo = ['User id', 'Day', 'EMA order', 'Stress lvl', 'Stress_label']

        feature_scaled = pd.DataFrame()
        try:
            scaler = MinMaxScaler()

            feature_df = preprocessed_df[StressModel.feature_df_with_state['features'].values]
            uinfo_df = preprocessed_df[userinfo].reset_index(drop=True)

            if norm_type == "default":
                # feature list
                feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

            elif norm_type == "new":

                feature_df = pd.concat([feature_df.reset_index(drop=True), new_row_preprocessed[
                    StressModel.feature_df_with_state['features'].values].reset_index(drop=True)])

                feature_scaled = pd.DataFrame(scaler.fit_transform(feature_df), columns=feature_df.columns)

                uinfo_df = uinfo_df.append({'User id': user_email, 'Day': day_num, 'EMA order': ema_order, 'Stress lvl': -1, 'Stress_label': -1}, ignore_index=True)

            feature_scaled = pd.concat([uinfo_df, feature_scaled.reset_index(drop=True)], axis=1)
        except Exception as e:
            print(e)

        return feature_scaled

    def initModel(self, norm_df):
        """
        initModel
        """
        print(".........initModel")

        try:
            print("===================Class Count :", Counter(norm_df['Stress_label']))
            # *** 만약 훈련 데이터에 0,1,2 라벨 중 하나가 없다면? 하나의 라벨만 존재한다면?

            X = norm_df[StressModel.feature_df_with_state['features'].values]
            Y = norm_df['Stress_label'].values

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

            model = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=100)
            # TODO split_num is 2 ==> related to stress prediction check(3)
            split_num = 2
            if len(Y) < split_num:
                split_num = len(Y_test)

            kfold = KFold(n_splits=split_num)
            scoring = {'accuracy': 'accuracy',
                       'f1_micro': 'f1_micro',
                       'f1_macro': 'f1_macro'}

            cv_results = cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring)
            model_result = {'accuracy': cv_results['test_accuracy'].mean(),
                            'f1_micro': cv_results['test_f1_micro'].mean(),
                            'f1_macro': cv_results['test_f1_macro'].mean()}

            print("===================Model Result : ", model_result)

            model.fit(X_train, Y_train)

            with open('model_result/' + str(self.uid) + "_model.p", 'wb') as file:
                pickle.dump(model, file)
                print("Model saved")


        except Exception as e:
            print(e)

    def saveAndGetSHAP(self, user_all_label, pred, new_row_raw, new_row_norm, initModel):

        model_results = []

        shap.initjs()
        explainer = shap.TreeExplainer(initModel)
        explainer.feature_perturbation = "tree_path_dependent"

        features = StressModel.feature_df_with_state['features'].values
        feature_state_df = StressModel.feature_df_with_state

        shap_values = explainer.shap_values(new_row_norm[features])
        #print(shap_values)
        expected_value = explainer.expected_value

        if len(expected_value) != len(user_all_label):
            with open('data_result/' + str(self.uid) + "_features.p", 'rb') as file:
                preprocessed = pickle.load(file)

            norm_df = StressModel.normalizing(self, "default", preprocessed, None, None, None, None)
            StressModel.initModel(self, norm_df)

            explainer = shap.TreeExplainer(initModel)
            explainer.feature_perturbation = "tree_path_dependent"

            features = StressModel.feature_df_with_state['features'].values
            feature_state_df = StressModel.feature_df_with_state

            shap_values = explainer.shap_values(new_row_norm[features], check_additivity = False)
            # print(shap_values)
            expected_value = explainer.expected_value


        try:
            ## changed 0723
            check_label = [0 for i in range(3)]

            for label in user_all_label:  # 유저한테 있는 Stress label 에 따라
                check_label[label] = 1
                feature_list = ""

                index = user_all_label.index(label)
                shap_accuracy = expected_value[index]
                shap_list = shap_values[index]

                shap_dict = dict(zip(features, shap_list[0]))
                shap_dict_sorted = sorted(shap_dict.items(), key=(lambda x: x[1]), reverse=True)

                act_features = ['Duration WALKING', 'Duration RUNNING', 'Duration BICYCLE', 'Duration ON_FOOT', 'Duration VEHICLE']
                app_features = ['Social & Communication','Entertainment & Music','Utilities','Shopping', 'Games & Comics', 'Health & Wellness', 'Education', 'Travel', 'Art & Design & Photo', 'News & Magazine', 'Food & Drink']

                act_tmp = ""
                for feature_name, s_value in shap_dict_sorted:
                    if s_value > 0:
                        feature_id = feature_state_df[feature_state_df['features'] == feature_name]['feature_id'].values[0]
                        feature_value = new_row_norm[feature_name].values[0]
                        if new_row_raw[feature_name].values[0] != 0:
                            if feature_name in act_features:
                                if act_tmp == "":
                                    act_tmp += feature_name

                                    if feature_value >= 0.5:
                                        feature_list += str(feature_id) + '-high '
                                    else:
                                        feature_list += str(feature_id) + '-low '
                            elif feature_name in app_features:
                                # Add package
                                try:
                                    pkg_result = AppUsed.objects.get(uid=self.uid, day_num=self.dayNo,
                                                                           ema_order=self.emaNo)
                                    pkg_text = ""
                                    if feature_name == "Entertainment & Music":
                                        pkg_text = pkg_result.Entertainment_Music
                                    elif feature_name == "Utilities":
                                        pkg_text = pkg_result.Utilities
                                    elif feature_name == "Shopping":
                                        pkg_text = pkg_result.Shopping
                                    elif feature_name == "Games & Comics":
                                        pkg_text = pkg_result.Games_Comics
                                    elif feature_name == "Others":
                                        pkg_text = pkg_result.Others
                                    elif feature_name == "Health & Wellness":
                                        pkg_text = pkg_result.Health_Wellness
                                    elif feature_name == "Social & Communication":
                                        pkg_text = pkg_result.Social_Communication
                                    elif feature_name == "Education":
                                        pkg_text = pkg_result.Education
                                    elif feature_name == "Travel":
                                        pkg_text = pkg_result.Travel
                                    elif feature_name == "Art & Design & Photo":
                                        pkg_text = pkg_result.Art_Photo
                                    elif feature_name == "News & Magazine":
                                        pkg_text = pkg_result.News_Magazine
                                    elif feature_name == "Food & Drink":
                                        pkg_text = pkg_result.Food_Drink

                                    if pkg_text != "":
                                        if feature_value >= 0.5:
                                            feature_list += str(feature_id) + '-high&' + pkg_text + " "
                                        else:
                                            feature_list += str(feature_id) + '-low '

                                except Exception as e:
                                    print("Exception during making feature_list of app...get AppUsed db", e)



                            else:
                                if feature_value >= 0.5:
                                    feature_list += str(feature_id) + '-high '
                                else:
                                    feature_list += str(feature_id) + '-low '

                if feature_list == "":
                    feature_list = "NO_FEATURES"

                if label == pred:
                    model_result = ModelResult.objects.create(uid=self.uid, day_num=self.dayNo, ema_order=self.emaNo,
                                                              prediction_result=label, accuracy=shap_accuracy,
                                                              feature_ids=feature_list, model_tag=True)
                else:
                    model_result = ModelResult.objects.create(uid=self.uid, day_num=self.dayNo, ema_order=self.emaNo,
                                                              prediction_result=label, accuracy=shap_accuracy,
                                                              feature_ids=feature_list)

                model_results.append(model_result)

            ## changed 0723
            ## For 문 끝난 후, model_result 에 없는 stress lvl 추가
            for i in range(3):
                if check_label[i] == 0:

                    if i == 0 : # LOW General message
                        feature_list = '0-general_0 7-general_0 12-general_0 18-general_0 29-general_0 '
                        model_result = ModelResult.objects.create(uid=self.uid, day_num=self.dayNo, ema_order=self.emaNo,
                                                                  prediction_result=i, accuracy=0,feature_ids=feature_list)
                    else: #LITTLE HIGH, HIGH General message
                        feature_list = '0-general_1 7-general_1 12-general_1 18-general_1 29-general_1 '
                        model_result = ModelResult.objects.create(uid=self.uid, day_num=self.dayNo,
                                                                  ema_order=self.emaNo,
                                                                  prediction_result=i, accuracy=0,
                                                                  feature_ids=feature_list)
                    model_results.append(model_result)


        except Exception as e:
            print("Exception at saveAndGetSHAP",e)

        return model_results

    def update(self, user_response, day_num, ema_order):
        # update Dataframe
        with open('data_result/' + str(self.uid) + "_features.p", 'rb') as file:
            preprocessed = pickle.load(file)

            # TODO : iloc 이나 다른 방법 사용해보기
            #preprocessed[(preprocessed['Day'] == day_num) and (preprocessed['EMA order'] == ema_order)]['Stress_label'] = user_response
            try:

                preprocessed.loc[(preprocessed['Day'] == day_num) & (preprocessed['EMA order'] == ema_order), 'Stress_label'] = user_response

                print("Changed!!!!!")

                with open('data_result/' + str(self.uid) + "_features.p", 'wb') as file:
                    pickle.dump(preprocessed, file)

            except Exception as e:
                print(e)
        # retrain the model
        norm_df = StressModel.normalizing(self, "default", preprocessed, None, None, None, None)
        StressModel.initModel(self, norm_df)

        # update ModelResult Table
        model_result = ModelResult.objects.get(uid=self.uid, day_num=day_num, ema_order=ema_order,
                                               prediction_result=user_response)
        model_result.user_tag = True
        model_result.save()
