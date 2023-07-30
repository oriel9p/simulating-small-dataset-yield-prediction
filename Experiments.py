import math

import pandas as pd
import numpy as np
from copy import copy
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier # AdaBoost
from sklearn.ensemble import AdaBoostRegressor # AdaBoost
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score,r2_score,mean_squared_error, mean_absolute_error,explained_variance_score,\
    precision_score, accuracy_score, recall_score, roc_curve, auc, precision_recall_curve, f1_score
import warnings
import torch
warnings.filterwarnings("ignore")

# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn import preprocessing
import matplotlib.pyplot as plt

# ---------- FUNCTIONS -------------- #


# Split data to train and test - v0.2
def data_split(cohort, train_cols, target_col, test_size, stratifier=None):
    X, y = cohort[train_cols], cohort[target_col]
    X_train, X_test, y_train, y_test = train_test_split \
        (X, y, test_size=test_size, stratify=stratifier, random_state=42)
    return X_train, X_test, y_train, y_test


# train prediction model
def fit_model(X_train, y_train, model='adaboost'):
    # Model object creation and fit
    if model == 'adaboost':
        model_fitted = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    elif model == 'adareg':
        model_fitted = AdaBoostRegressor(n_estimators=100, random_state=42).fit(X_train,y_train)
    elif model == 'lr':
        model_fitted = LogisticRegression(random_state=42).fit(X_train,y_train)
    else:
        raise ValueError('Model type not in list')
    return (model_fitted)


# model evaluation
def model_evaluate(model, X_test, y_test, eval_criteria='roc_auc'):
    e_metric = 0
    if eval_criteria == 'roc_auc':
        e_metric = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    elif eval_criteria == 'precision':
        e_metric = precision_score(y_test, model.predict(X_test))
    elif eval_criteria == 'accuracy':
        e_metric = accuracy_score(y_test, model.predict(X_test))
    elif eval_criteria == 'recall':
        e_metric = recall_score(y_test, model.predict(X_test))
    elif eval_criteria == 'pr_auc':
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        e_metric = auc(recall, precision)
    elif eval_criteria == 'f1':
        e_metric = f1_score(y_test, model.predict(X_test))
    elif eval_criteria == 'R2':
        e_metric = r2_score(y_test, model.predict(X_test))
    elif eval_criteria == 'MAE':
        e_metric = mean_absolute_error(y_test, model.predict(X_test))
    elif eval_criteria == 'MSE':
        e_metric = mean_squared_error(y_test, model.predict(X_test))

    elif eval_criteria == 'ev':
        e_metric = explained_variance_score(y_test, model.predict(X_test))
    else:
        raise ValueError('Evaluation criteria not in list')
    return e_metric


def train_generator(cohort,metadata, discrete_columns, epochs, batch_size, X_test, y_test, label, params=None):
    # original usage
    # generator_model = CTGANSynthesizer(epochs=300, batch_size=250)
    generator_model = CTGANSynthesizer(
         epochs=epochs, batch_size=batch_size,verbose=True, metadata=metadata, X_test=X_test, y_test=y_test, label=label
    )
    generator_model.fit(cohort)
    return generator_model


def generate_samples_binary(generator_model, n_samples, label, prevalence):
    # Stratify sampling to ensure same prevalence as the original set
    # sample positive class and negative class
    condition_pos = Condition({label:1}, num_rows=n_samples * prevalence)
    condition_neg = Condition({label:0}, num_rows=n_samples * (1-prevalence))
    # Synthetic copy generation
    samples_pos = generator_model.sample_from_conditions(conditions=[condition_pos])
    samples_neg = generator_model.sample_from_conditions(conditions=[condition_neg])
    samples = pd.concat([samples_pos, samples_neg])
    return samples

def import_n_setup(use_case):
    if use_case=='heart':
        data = pd.read_csv('datasets/heart_fail.csv')
        discrete_columns = ['Sex','ChestPainType','RestingECG',
                            'ExerciseAngina','ST_Slope','HeartDisease']
        label = 'HeartDisease'
    elif use_case=='adult':
        data = pd.read_csv('datasets/adults_concesus_income.csv')
        discrete_columns = ['workclass','education','marital-status',
                            'occupation','relationship','race','sex','native-country', 'Target']
        label = 'target'
    elif use_case=='students':
        data = pd.read_csv('datasets/students_dropout.csv')
        discrete_columns = ['Marital status', 'Application order', 'Attendance',
                            'Previous qualification', 'Mom Qualification', 'Dad Qualification',
                            'Mom Occupation', 'Dad Occupation', 'Educational special needs','Debtor','Gender',
                            'Scholarship holder', 'Target']
        label = 'Target'
    elif use_case=='crop yield':
        data = pd.read_csv('datasets/crop_yield.csv')
        discrete_columns = ['Year','Item','Area']
        label = 'hg/ha_yield'
    elif use_case == 'fraud':
        data = pd.read_csv('datasets/cc_frauds.csv')
        discrete_columns = ['Class']
        label = 'Class'
        # Under sample for size and prevalence
        rus = RandomUnderSampler(random_state=42, sampling_strategy=0.05)
        cols = data.columns
        X, y = data.loc[:, data.columns != 'Class'], data['Class']
        X_res, y_res = rus.fit_resample(X,y)
        # RUS data, concat X,y
        data = pd.concat([X_res, y_res],axis=1)
    elif use_case == 'news':
        data = pd.read_csv('datasets/news_proper_data.csv')
        discrete_columns = ['publish_day','data_channel']
        label = 'shares'
    elif use_case == 'house':
        data = pd.read_csv('datasets/housr_price.csv')
        discrete_columns = ['ocean_proximity']
        label = 'median_house_value'
        data = data.dropna(axis=1)
    else:
        ValueError('No such dataset available')
        return None

    return data, discrete_columns, label
if __name__ == '__main__':
    # check for GPU
    print(torch.cuda.is_available())
    # results_dict = {'dataset': [], 'n': [], 'accuracy': [], 'syn_accuracy': [], 'f1': [],
    #                 'syn_f1': []}  # 'precision': [], 'recall': [], 'accuracy': []}
    #
    # ucs = ['fraud','heart','students','adult']
    # for use_case in ucs:
    #     data, discrete_columns, label = import_n_setup(use_case)
    #     print(data.shape[0])
    #     results_dict['dataset'].append(use_case)
    #     results_dict['n'].append(data.shape[0])
    #     # Train an XGBOOST
    #     X, y = data.loc[:, data.columns != label], data[label]
    #     X_train, X_test, y_train, y_test = train_test_split \
    #         (X, y, test_size=0.2, stratify=data[label], random_state=42)
    #
    #     ada_model = fit_model(X_train, y_train)
    #     accuracy = model_evaluate(ada_model,X_test, y_test, eval_criteria='accuracy')
    #     f1 = model_evaluate(ada_model, X_test, y_test, eval_criteria='f1')
    #     results_dict['accuracy'].append(accuracy)
    #     results_dict['f1'].append(f1)
    #
    #     # generate synthetic data
    #     train_data = X_train
    #     train_data[label] = y_train
    #     metadata = SingleTableMetadata()
    #     metadata.detect_from_dataframe(data=data)
    #     epochs = 3
    #     batch_size = 1000 if ucs == 'adult' else 300
    #     ctgan_model = train_generator(train_data,metadata,discrete_columns, epochs, batch_size, X_test, y_test, label) # TODO: added, X_test set, y_test set and label column name
    #     syn_sample_size = X_train.shape[0] if ucs == 'heart' else X_train.shape[0] // 2
    #     original_size = X_train.shape[0]
    #     samples = ctgan_model.sample(original_size) # sample equal to training set size
    #     # TODO: Stratify sampling to ensure same prevalence
    #     # prevalence = sum(data[label].tolist()) // len(data[label].tolist()) # positive class//all samples
    #     # samples = generate_samples_binary(ctgan_model, syn_sample_size, label, prevalence) # generate synthetic samples, relative to the original sample size
    #
    #
    #     # train on synthetic data
    #     sX, sy = samples.loc[:, samples.columns != label], samples[label]
    #
    #     ada_model = fit_model(sX, sy)
    #     syn_accuracy = model_evaluate(ada_model, X_test, y_test, eval_criteria='accuracy')
    #     syn_f1 = model_evaluate(ada_model, X_test, y_test, eval_criteria='f1')
    #     results_dict['syn_accuracy'].append(syn_accuracy)
    #     results_dict['syn_f1'].append(syn_f1)
    #     df = pd.DataFrame(results_dict)
    # df.to_csv('results_news_2307_feedback.csv')

    # regression run

    results_dict = {'dataset': [], 'n': [], 'RMSE': [], 'syn_RMSE': [], 'R2': [],
                    'syn_R2': []}  # 'precision': [], 'recall': [], 'accuracy': []}

    ucs = ['house']
    for use_case in ucs:
        data, discrete_columns, label = import_n_setup(use_case)
        print(data.shape[0])
        results_dict['dataset'].append(use_case)
        results_dict['n'].append(data.shape[0])
        # Train an XGBOOST
        X, y = data.loc[:, data.columns != label], data[label]
        X_train, X_test, y_train, y_test = train_test_split \
            (X, y, test_size=0.2, random_state=42)

        ada_model = fit_model(X_train, y_train, "adareg")
        MSE = model_evaluate(ada_model, X_test, y_test, eval_criteria='MSE')
        ev = model_evaluate(ada_model, X_test, y_test, eval_criteria='R2')
        results_dict['RMSE'].append(math.pow(MSE,0.5))
        results_dict['R2'].append(ev)

        # generate synthetic data
        train_data = X_train
        train_data[label] = y_train
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        epochs = 150
        batch_size = 2000 if use_case == 'house' else 100
        ctgan_model = train_generator(train_data, metadata, discrete_columns, epochs, batch_size, X_test, y_test,
                                      label)  # TODO: added, X_test set, y_test set and label column name
        syn_sample_size = X_train.shape[0] #if ucs == 'heart' else X_train.shape[0] // 2
        original_size = X_train.shape[0]
        samples = ctgan_model.sample(original_size)  # sample equal to training set size
        # TODO: Stratify sampling to ensure same prevalence
        # prevalence = sum(data[label].tolist()) // len(data[label].tolist()) # positive class//all samples
        # samples = generate_samples_binary(ctgan_model, syn_sample_size, label, prevalence) # generate synthetic samples, relative to the original sample size
        # join syn samples and real samples
        #samples = pd.concat([samples, train_data])
        # train on synthetic data
        sX, sy = samples.loc[:, samples.columns != label], samples[label]

        ada_model = fit_model(sX, sy,"adareg")
        syn_MSE = model_evaluate(ada_model, X_test, y_test, eval_criteria='MSE')
        syn_ev = model_evaluate(ada_model, X_test, y_test, eval_criteria='R2')
        results_dict['syn_RMSE'].append(math.pow(syn_MSE,0.5))
        results_dict['syn_R2'].append(syn_ev)
        df = pd.DataFrame(results_dict)
    df.to_csv('results_yieldfull_3007.csv')


