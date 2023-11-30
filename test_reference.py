from PIL import Image
import shutil
import pandas as pd
import numpy as np
import os
import random
import time
import copy
import cv2
import csv
import matplotlib.pyplot as plt

import xgboost
from xgboost import XGBRegressor
from xgboost import plot_importance, plot_tree

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# 데이터 받아오기
data = pd.read_excel('../MoldingSand2/Data/Preprocessing/1and2_remove.xlsx',engine='openpyxl',sheet_name='Sheet3')

# 데이터 추출
df = data[['GFD-6', 'GFD-12', 'GFD-20', 'GFD-30', 'GFD-40', 'GFD-50', 'GFD-70',
           'GFD-100', 'GFD-140', 'GFD-200', 'GFD-270', 'Pan', 'Moisture (% by wt)',
           'Percent M.B. Active Clay', 'AFS Clay Content (% <20 µm)', 'AFS Grain Fineness No.',
           'Compactability (%)', 'Green Comp. Strength (N/sq cm)', 'Permeability No.']]

# column 이름 바꾸기
df.columns = [['GFD-6', 'GFD-12', 'GFD-20', 'GFD-30', 'GFD-40', 'GFD-50', 'GFD-70',
                 'GFD-100', 'GFD-140', 'GFD-200', 'GFD-270', 'Pan', 'Moisture',
                'Active Clay', 'ASF Clay Content', 'AFS Grain Fineness No.',
                'Compactability', 'Green Comp. Strength', 'Permeability No.']]

df.head()

data_0 = np.squeeze(df[['GFD-6']].to_numpy()) # 모두 0이라서 넣어줄 필요 없을듯
data_1 = np.squeeze(df[['GFD-12']].to_numpy())
data_2 = np.squeeze(df[['GFD-20']].to_numpy())
data_3 = np.squeeze(df[['GFD-30']].to_numpy())
data_4 = np.squeeze(df[['GFD-40']].to_numpy())
data_5 = np.squeeze(df[['GFD-50']].to_numpy())
data_6 = np.squeeze(df[['GFD-70']].to_numpy())
data_7 = np.squeeze(df[['GFD-100']].to_numpy())
data_8 = np.squeeze(df[['GFD-140']].to_numpy())
data_9 = np.squeeze(df[['GFD-200']].to_numpy())
data_10 = np.squeeze(df[['GFD-270']].to_numpy())
data_11 = np.squeeze(df[['Pan']].to_numpy())
data_12 = np.squeeze(df[['Moisture']].to_numpy()) # 12 - 13 같이 들어가니까 Loss가 더 낮아짐
data_13 = np.squeeze(df[['Active Clay']].to_numpy()) # Loss를 낮추는데 역할을 한다!
data_14 = np.squeeze(df[['ASF Clay Content']].to_numpy())
data_15 = np.squeeze(df[['AFS Grain Fineness No.']].to_numpy())
data_16 = np.squeeze(df[['Compactability']].to_numpy())
data_17 = np.squeeze(df[['Green Comp. Strength']].to_numpy())
data_18 = np.squeeze(df[['Permeability No.']].to_numpy())
######################################################################################

all_data_x = np.zeros((len(data_3),13))
all_data_y = np.zeros((len(data_13),))

# all_data_x[:,0] = data_0
# all_data_x[:,1] = data_1
# all_data_x[:,2] = data_2
# all_data_x[:,3] = data_3
# all_data_x[:,4] = data_4
# all_data_x[:,5] = data_5
# all_data_x[:,6] = data_6
# all_data_x[:,7] = data_7
# all_data_x[:,8] = data_8
# all_data_x[:,9] = data_9
# all_data_x[:,10] = data_10
# all_data_x[:,11] = data_11
# all_data_x[:,12] = data_12
# all_data_x[:,13] = data_13
# all_data_x[:,14] = data_14
# all_data_x[:,15] = data_15

# OUTPUT 1 성능 젤 잘나오는 애 
# all_data_x[:,0] = data_5
# all_data_x[:,1] = data_6
# all_data_x[:,2] = data_7
# all_data_x[:,3] = data_8
# all_data_x[:,4] = data_10
# all_data_x[:,5] = data_11
# all_data_x[:,6] = data_12
# all_data_x[:,7] = data_13
# all_data_x[:,8] = data_14
# all_data_x[:,9] = data_15
##############################################

# all_data_x[:,0] = data_3
# all_data_x[:,1] = data_5
# all_data_x[:,2] = data_6
# all_data_x[:,3] = data_7
# all_data_x[:,4] = data_10
# all_data_x[:,5] = data_11
# all_data_x[:,6] = data_12
# all_data_x[:,7] = data_13
# all_data_x[:,8] = data_14
# all_data_x[:,9] = data_15

# all_data_x[:,0] = data_3
# all_data_x[:,1] = data_5
# all_data_x[:,2] = data_6
# all_data_x[:,3] = data_7
# all_data_x[:,4] = data_9
# all_data_x[:,5] = data_10
# all_data_x[:,6] = data_11
# all_data_x[:,7] = data_12
# all_data_x[:,8] = data_13
# all_data_x[:,9] = data_14
# all_data_x[:,10] = data_15

# all_data_x[:,0] = data_3
# all_data_x[:,1] = data_4
# all_data_x[:,2] = data_5
# all_data_x[:,3] = data_6
# all_data_x[:,4] = data_7
# all_data_x[:,5] = data_8
# all_data_x[:,6] = data_9
# all_data_x[:,7] = data_10
# all_data_x[:,8] = data_11
# all_data_x[:,9] = data_12
# all_data_x[:,10] = data_13
# all_data_x[:,11] = data_14
# all_data_x[:,12] = data_15

#########################################################
# output 2 12,20,40,70 뺀거
# all_data_x[:,0] = data_3
# all_data_x[:,1] = data_5
# all_data_x[:,2] = data_7
# all_data_x[:,3] = data_8
# all_data_x[:,4] = data_9
# all_data_x[:,5] = data_10
# all_data_x[:,6] = data_11
# all_data_x[:,7] = data_12
# all_data_x[:,8] = data_13
# all_data_x[:,9] = data_14
# all_data_x[:,10] = data_15

# output 2 12,20,40,70,50,30 뺀거
# all_data_x[:,0] = data_7
# all_data_x[:,1] = data_8
# all_data_x[:,2] = data_9
# all_data_x[:,3] = data_10
# all_data_x[:,4] = data_11
# all_data_x[:,5] = data_12
# all_data_x[:,6] = data_13
# all_data_x[:,7] = data_14
# all_data_x[:,8] = data_15

# output 2 12,20,40,70,50,30,100 뺀거

# all_data_x[:,0] = data_8
# all_data_x[:,1] = data_9
# all_data_x[:,2] = data_10
# all_data_x[:,3] = data_11
# all_data_x[:,4] = data_12
# all_data_x[:,5] = data_13
# all_data_x[:,6] = data_14
# all_data_x[:,7] = data_15

# output 2 12,20,40,70,50,30,100,270 뺀거

# all_data_x[:,0] = data_8
# all_data_x[:,1] = data_9
# all_data_x[:,2] = data_11
# all_data_x[:,3] = data_12
# all_data_x[:,4] = data_13
# all_data_x[:,5] = data_14
# all_data_x[:,6] = data_15

###########################################################
# all_data_x[:,0] = data_1
# all_data_x[:,1] = data_2
# all_data_x[:,2] = data_3
# all_data_x[:,3] = data_4
# all_data_x[:,4] = data_5
# all_data_x[:,5] = data_6
# all_data_x[:,6] = data_7
# all_data_x[:,7] = data_8
# all_data_x[:,8] = data_9
# all_data_x[:,9] = data_10
# all_data_x[:,10] = data_11
# all_data_x[:,11] = data_12
# all_data_x[:,12] = data_13
# all_data_x[:,13] = data_14
# all_data_x[:,14] = data_15

all_data_x[:,0] = data_3
all_data_x[:,1] = data_4
all_data_x[:,2] = data_5
all_data_x[:,3] = data_6
all_data_x[:,4] = data_7
all_data_x[:,5] = data_8
all_data_x[:,6] = data_9
all_data_x[:,7] = data_10
all_data_x[:,8] = data_11
all_data_x[:,9] = data_12
all_data_x[:,10] = data_13
all_data_x[:,11] = data_14
all_data_x[:,12] = data_15

all_data_y = data_18 # 16, 17 ,18

np.random.seed(5)

index = np.arange(0, all_data_x.shape[0])
np.random.shuffle(index)
all_data_x = all_data_x[index]
all_data_y = all_data_y[index]

######################################################################################
print(all_data_x.shape)
print(all_data_y.shape)

x_train, x_test, y_train, y_test = train_test_split(all_data_x, all_data_y, shuffle = False, test_size = 0.2) 

print(x_train.shape) 
print(type(x_train)) 
print(y_train.shape) 
print(type(y_train)) 

mean_x = np.mean(x_train, axis = 0)   
std_x = np.std(x_train, axis = 0)   

mean_y = np.mean(y_train, axis = 0)   
std_y = np.std(y_train, axis = 0)   

x_train = (x_train - mean_x) / std_x
# y_train = (y_train - mean_y) / std_y

x_test = (x_test - mean_x) / std_x

n_estimators = [150 ,200, 250, 300, 350, 400, 450, 500, 550, 600]
learning_rate = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
max_depth = [2, 3, 4, 5, 6, 7 ,8, 9, 10]
subsample = [0.2, 0.4, 0.6, 0.8, 1.0]

f = open('../MoldingSand2/XGBoost.csv', 'w', newline ='')
wr = csv.writer(f)
wr.writerow(['n_estimators','learning_rate','max_depth','subsample','MAE'])

for a in range (len(n_estimators)):
    for b in range (len(learning_rate)):
        for c in range (len(max_depth)):
            for d in range (len(subsample)):
                model = XGBRegressor(n_estimators = n_estimators[a],
                                     learning_rate = learning_rate[b],
                                     max_depth = max_depth[c],
                                     subsample = subsample[d]
                                     )
                model.fit(x_train, y_train) # train
                
                y_pred = model.predict(x_test) # test
                y_true = y_test
                mae = mean_absolute_error(y_true, y_pred)
                
                wr.writerow([n_estimators[a], learning_rate[b], max_depth[c], subsample[d], mae])
f.close()

model = XGBRegressor()
model_param_grid = {'n_estimators' : [200, 250, 300, 350, 400, 450],
                    'learning_rate' : [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth' : [3, 4, 5, 6, 7 ,8],
                    'subsample': [0.6, 0.8, 1.0]
                   }

model_grid = GridSearchCV(model,
                          param_grid = model_param_grid,
                          scoring = 'neg_mean_absolute_error', #make_score(mean_absolute_error, squared=False),
                          n_jobs = -1,
                          verbose = 2,
                          cv = 4,
                          refit = True
                         )
model_grid.fit(x_train, y_train)#, **{'eval_metric': 'mae'})
# print('GridSearchCV mean(mae)?:  ', model_grid.cv_results_['mean_test_score'])

# model = XGBRegressor(n_estimators=200, learning_rate = 0.05, subsample = 0.6) # 트리의 개수 300개로 모델 생성
# model.fit(x_train, y_train)
# model
# # objective='reg:linear'

model_grid_df = pd.DataFrame(model_grid.cv_results_)
model_grid_df.loc[:, ['mean_test_score', "params"]]

for i in range(648):
    if model_grid_df.loc[:, ['rank_test_score', "params"]]['rank_test_score'][i] == 1:
        print(model_grid_df.loc[:, ['rank_test_score', "params"]]['params'][i])


###########################16########################
# model_test = XGBRegressor(learning_rate = 0.1,
#                           max_depth = 4,
#                           n_estimators = 200,
#                           subsample = 0.6
#                          )

# model_test = XGBRegressor(learning_rate = 0.05,
#                           max_depth = 5,
#                           n_estimators = 450,
#                           subsample = 1
#                          )
###########################17########################
# model_test = XGBRegressor(learning_rate = 0.1,
#                           max_depth = 6,
#                           n_estimators = 400,
#                           subsample = 0.6
#                          )

# model_test = XGBRegressor(learning_rate = 0.05,
#                           max_depth = 3,
#                           n_estimators = 200,
#                           subsample = 0.6
#                          )

###########################18########################
# model_test = XGBRegressor(learning_rate = 0.1,
#                           max_depth = 7,
#                           n_estimators = 200,
#                           subsample = 0.6
#                          )
model_test = XGBRegressor(learning_rate = 0.05,
                          max_depth = 5,
                          n_estimators = 250,
                          subsample = 0.6
                         )

# model_test = XGBRegressor(n_estimators = 300)

model_test.fit(x_train, y_train)

plot_importance(model_test)
# plot_importance(model)

y_pred = model_test.predict(x_test)
y_true = y_test

# y_pred = (y_pred * std_y) + mean_y 

mae = mean_absolute_error(y_true, y_pred) 
# mse = mean_squared_error(y_true, y_pred) 
# print('mse: ', mse)
print('mae: ', mae) 

fig = plt.figure( figsize = (12, 4) )
chart = fig.add_subplot(1,1,1)
chart.plot(y_true, marker='o', color='blue', label='real value')
chart.plot(y_pred, marker='^', color='red', label='predict value')
chart.set_title('real value vs predict value')
plt.xlabel('index')
plt.ylabel('real vs predict')
plt.legend(loc = 'best')