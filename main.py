
import numpy as np
import pandas as pd
import datetime

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.preprocessing import MinMaxScaler

from model.feature_select import fnFeatSelect_RandVar
from model.optimize_hyperparameter import fnOpt_HyperPara
from utils.metrics import RMSE
from utils.metrics import MAPE

from hyperopt import fmin, hp, STATUS_OK, Trials, tpe

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")


####################################################################################
#### Data Load

## Sample Data
from sklearn.datasets import fetch_california_housing

## Loader
housing = fetch_california_housing()
## Feature Data
dataX = pd.DataFrame(
    housing.data,
    columns = housing.feature_names
)
## Target Data
dataY = pd.DataFrame(
    housing.target,
    columns = housing.target_names
)
## Union: Total Data
totalData = pd.concat([dataX, dataY], axis = 1)

print('Data: {}'.format(totalData.shape))
print('Feature Types')
print(totalData.dtypes, '\n')

featureList = dataX.columns.tolist()
TargetNM = housing.target_names[0]


####################################################################################
#### Train / Test 분할(8:2)

## 학습Data Index
TRAIN_CNT = int(len(totalData) * 0.8)
trainData = totalData[:TRAIN_CNT]
testData = totalData[TRAIN_CNT:]

print('Train Data: {}'.format(trainData.shape))
print('Test Data: {}'.format(testData.shape))


####################################################################################
#### 전처리

## 변수 Scale
scaler = MinMaxScaler()
trainScale = scaler.fit_transform(trainData[featureList])
testScale = scaler.transform(testData[featureList])

## DataFrame 변환
trainScale_DF = pd.DataFrame(
    trainScale,
    columns = featureList
)
trainScale_DF[TargetNM] = trainData[TargetNM].values
testScale_DF = pd.DataFrame(
    testScale,
    columns = featureList
)
testScale_DF[TargetNM] = testData[TargetNM].values

print('Scaled Train Info')
print(trainScale_DF.describe(), '\n')
print('Scaled Test Info')
print(testScale_DF.describe(), '\n')


####################################################################################
#### 변수선택

## 임의로 생성한 Random변수들보다 변수중요도가 떨어지는 변수들을 제거하는 로직
feat_RandSelct_LS = fnFeatSelect_RandVar(
    df_x = trainScale_DF[featureList], 
    df_y = trainScale_DF[TargetNM], 
    x_var = featureList, 
    core_cnt = -1
)
deleteFeatureLS = list(set(featureList) - set(feat_RandSelct_LS))
print('Feauter List with Rand-Select: {}'.format(feat_RandSelct_LS))

if len(deleteFeatureLS) > 0:
    print('Delete Feature List: {}'.format(deleteFeatureLS))

## 최종 변수 리스트
feat_Final_LS = feat_RandSelct_LS


####################################################################################
#### Hyper-parameter 최적화

MODEL_NM = 'xgb'
H_PARA_SPACE = {
    'max_depth': hp.uniform("max_depth", 1, 30),
    'min_child_weight': hp.loguniform('min_child_weight', -3, 3),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.loguniform('gamma', -10, 10)
}

TrialResult, BestPara = fnOpt_HyperPara(
    total_data = trainScale_DF, 
    x_var = feat_Final_LS, 
    y_var = TargetNM, 
    space = H_PARA_SPACE, 
    lean_rate_ls = [0.001, 0.01, 0.1], 
    ml_model = MODEL_NM, 
    core_cnt = -1, 
    cv_num = 3, 
    max_evals = 30, 
    seed = 1000, 
    verbose = True,
)


####################################################################################
#### 최종 모델학습 및 예측
finalModel = XGBRegressor(**BestPara)
finalModel.fit(
    trainScale_DF[feat_Final_LS], 
    trainScale_DF[TargetNM],
    early_stopping_rounds = 50,
    eval_metric = 'rmse',
    eval_set = [(trainScale_DF[feat_Final_LS], trainScale_DF[TargetNM])],
    verbose = False
)

## Predict
finalPredict = finalModel.predict(testScale_DF[feat_Final_LS])
## Compare Predict vs Real
finalPredict_DF = pd.DataFrame({
    'PRED': finalPredict,
    'REAL': testScale_DF[TargetNM]
    }
)

## Score
SCORE_RMSE = RMSE(
    pred = finalPredict,
    true = testScale_DF[TargetNM].values
)
SCORE_MAPE = MAPE(
    pred = finalPredict,
    true = testScale_DF[TargetNM].values
)
print("RMSE: {}".format(SCORE_RMSE))
print("MAPE: {}".format(SCORE_MAPE))

## Plotting
plt.scatter(
    finalPredict_DF['REAL'],
    finalPredict_DF['PRED']
)
plt.title(f'{MODEL_NM.upper()}: Real vs Predict')
plt.xlabel(f'Real {TargetNM}')
plt.ylabel(f'Predict {TargetNM}')

plt.plot(
    np.arange(int(finalPredict_DF['REAL'].min()), int(finalPredict_DF['REAL'].max())),
    np.arange(int(finalPredict_DF['REAL'].min()), int(finalPredict_DF['REAL'].max())),
    linestyle = '--',
    color = 'red'
)
plt.show()
