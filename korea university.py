from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


sns.get_dataset_names()



iris = sns.load_dataset('iris')

iris['species'].value_counts().plot(kind='bar')

X = iris.drop('species', axis=1)

sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species')



X = X.values # Series, DataFrame 형태를 numpy array 변경하기



le = LabelEncoder()
le.fit_transform(y) # y값에 대해 fit_transform 함수 이용해서 라벨인코딩 수행



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , stratify=y, random_state=42)



dt = DecisionTreeClassifier() / rf = RandomForestClassifier() #모델 정의
dt.fit(X_train, y_train) # 모델 학습
dt.score(X_test, y_test) # 모델 성능확인



pred = rf.predict(X_test[0:1]) # X_test 첫라인 샘플 데이터을 모델 입력해서 예측하기



model = Sequential()
model.add(Dense(6, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.title('Loss and Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Accuracy", "Val_loss", "Val_Accuracy"])
plt.show()



tips = sns.load_dataset('tips')

tips.sex.replace(['Female', 'Male'], [0,1], inplace=True)
tips.time.replace(['Dinner', 'Lunch'], [1,0], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
tips.day = le.fit_transform(tips.day)

tips.to_csv('tips2.csv', index=False)



X = tips2.drop('tip', axis=1)



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )



model = Sequential()
model.add(  Dense( 4, activation='relu', input_shape=(6,)  )  )
model.add(  Dense( 4, activation='relu',  )  )
model.add(  Dense( 1, activation=None  )  )



# pred 결과에 대한 분류값 출력 : np.argmax(pred, axis=1)

pred = model.predict(X_test[0:2])

np.argmax(pred, axis=1)



# 모델 컴파일
# loss='mse', optimizer='adam', metrics=['accuracy']
# loss='mse', optimizer='adam', metrics=['mse', 'mae']

model.compile(  loss='mse', optimizer='adam', metrics=['accuracy']  )
model.compile(  loss='mse', optimizer='adam', metrics=['mse', 'mae']  )



# EarlyStopping : 매 epoch 마다 'val_loss' 측정해서  3번 동안 더 이상 성능이 나아지지 않으면 조기 종료
# onitor='val_loss' , patience=3

es = EarlyStopping( monitor='val_loss' , patience=3 , verbose=1 )



# ModelCheckpoint : 매 epoch 마다 'val_loss' 측정해서  이전보다 성능이 더 좋아지면 모델을 저장
# best_model.h5 이름으로 저장
# monitor='val_loss', save_best_only=True

mc = ModelCheckpoint( 'best_model.h5', monitor='val_loss', save_best_only=True, berbose=1  )



# 모델 학습(fit)
# - X, y : X_train , y_train
# - epochs=50,
# - batch_size=8
# - validation_data=(X_test, y_test)
# - callbacks=[es, mc]
# - 모델 학습 결과 : history 저장

history = model.fit(  X_train , y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test) , callbacks=[es, mc] )



# DNN 모델의 predict함수를  활용해서 시뮬레이션 데이터 예측하기

simul = [3.23, 1. , 0. , 2. , 1. , 2. ]
model.predict([ simul ])



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


tips = sns.load_dataset('tips')

tips.sex.replace(['Female', 'Male'], [0,1], inplace=True)
tips.smoker.replace(['No', 'Yes'], [0,1], inplace=True)
tips.time.replace(['Dinner', 'Lunch'], [1,0], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
tips.day = le.fit_transform(tips.day)

tips.to_csv('tips2.csv', index=False)



# 학습 데이터셋과 테스트 데이터셋 나누기
# train_test_split 함수 활용 >> 8:2 비율로 학습셋과 테스트셋 나누기, random_state=41
# 학습셋과 테스트셋 shape 확인 : X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=41 )
X_train.shape, X_test.shape, y_train.shape, y_test.shape



# 1. RandomForestRegressor 모델 정의 : 트리의 갯수 50, 트리의 깊이 6 -> rfr 저장
# 2. rfr 모델 학습 : X_train, y_train
# 3. rfr 모델 성능확인 : X_test, y_test

rfr = RandomForestRegressor( n_estimators=50, max_depth=6 )
rfr.fit(  X_train, y_train  )
rfr.score(  X_test, y_test  )



# 테스트셋의 0 라인 샘플 데이터을 모델 입력해서 예측하기
# rfr 모델의 predict 함수 활용
# 입력 : [X_test[0]], 결과 : pred 저장
# pred 결과 출력

pred = rfr.predict([X_test[0]])
print(pred)



# 1. DecisionTreeClassifier 모델 정의 -> dtc 저장
# 2. dtc 모델 학습 : X_train, y_train
# 3. dtc 모델 성능확인 : X_test, y_test

dtc = DecisionTreeClassifier()
dtc.fit( X_train, y_train )
dtc.score( X_test, y_test )



# 1. RandomForestClassifier 모델 정의 -> rfc 저장
# 2. rfc 모델 학습 : X_train, y_train
# 3. rfc 모델 성능확인 : X_test, y_test

rfc = RandomForestClassifier()
rfc.fit( X_train, y_train )
rfc.score( X_test, y_test )



# sklearn.neighbors 밑에 KNeighborsRegressor 임포트 및 모델 학습
# KNeighborsRegressor 모델 정의시 n_neighbors 이웃갯수 : 3 으로 설정

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)



# KNN 모델의 predict 함수를 사용하여 시뮬레이션 데이터를 예측하기

simul = [3.23, 1. , 0. , 2. , 1. , 2. ]
knn.predict([ simul ])



cust_info = pd.read_csv('cust_info.csv')
service_info = pd.read_csv('service_info.csv')

df = pd.merge(cust_info, service_info, on='customerID')



df.head()
df.tail()
df.info()
df.index
df.columns
df.values
df.isnull()
df.describe()



df.drop('customerID', axis=1, inplace=True)



# TotalCharges 컬럼 타입을 float로 변경해 보자.
# 문자열을 숫자형으로 변경할수 없으므로 에러 발생

df['TotalCharges'].astype(float)



# TotalCharges 컬럼 값들중에 ' ' 공백 있는지 확인해 보자
# Boolean indexing으로 검색
# Boolean indexing에서 2개 이상의 조건을 연결시 AND : & , OR : |  구분하여 사용해 줘야 한다.

(df['TotalCharges'] == '') | (df['TotalCharges'] == ' ')



df['TotalCharges'].replace([' '], ['0'], inplace=True)
df['Churn'].replace(['Yes', 'No'], [1, 0], inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype(float)



df['Churn'].value_counts()



# 1. 결측치 많은 DeviceProtection 컬럼 제거  --> drop 함수 활용
# 2. Dependents 컬럼의 결측치 : 최빈값 채우기  --> fillna 함수, mode 함수 활용
# 3. MonthlyCharges 컬럼의 결측치 : 평균값 채우기  --> fillna 함수, mean 함수 활용
# 4. 나머지 결측치 제거 : dropna 함수 활용


df.drop( 'DeviceProtection', axis=1, inplace=True)
df['Dependents'] = df['Dependents'].fillna( df['Dependents'].mode() )
df['MonthlyCharges'] =  df['MonthlyCharges'].fillna( df['MonthlyCharges'].mean() )
df.dropna(inplace=True)



# 1. 숫자형 컬럼 추출하기 : select_dtypes() 함수 활용, 옵션으로 'number' 혹은 ['int', 'float']
# 2. df 데이터프레임에서 숫자형 추출하고 'Churn' 컬럼을 기준으로 그룹화 및 평균 출력하기, Churn : 0 이탈X, 1 이탈

df.select_dtypes(['int', 'float']).head()
df.select_dtypes(['int', 'float']).groupby(['Churn']).mean()



# 'gender' , 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn' 컬럼을 대상으로
# Churn + gender 그룹화 및 평균 출력

df[ ['gender' , 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'] ].groupby([ 'Churn' , 'gender' ]).mean()



df['gender'].value_counts().plot(kind='bar')



# Object 컬럼만 뽑으
# number(int, float) 컬럼에 대해 검색

df.select_dtypes('O').head(3)
df.select_dtypes( 'number').head(3)



# Object 컬럼 하나씩 가져와서 Bar 차트 그려보기

object_list = df.select_dtypes('object').columns.values

for col in object_list:
    df[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()



# matplotlib 활용해서 tenure 컬럼의 히스토그램을 그려보자.

plt.subplot(131)
df['tenure'].plot(kind='hist') # 방법1

plt.subplot(133)
plt.hist(df['tenure']) # 방법2
plt.show()



# seaborn의 histplot 이용해서 tenure (서비스 사용기간)에 대한 히스토그램

# seaborn kdeplot : 히스토그램을 곡선으로 그려보자. (histplot 비슷)

# seaborn countplot 함수 이용해서 MultipleLines 서비스에 대한 갯수 분포 확인하기(hue='churn' 이용)

# seaborn plot 옵션 대략 형태 : data=df, x='aaa', y='bbb', hue='ccc'
# 히스토그램 분석 : 처음에 많이 사용하고 , 70개월 사용하는 충성고객도 있다.
# seaborn의 histplot 이용해서 tenure (서비스 사용기간) 대한 히스토그램을 Churn 으로 구분
# 히스토그램으로 Churn 구분하니 겹쳐서 보기 어렵다.

sns.histplot(data=df, x='tenure', hue='Churn')



# 추가 : 기존 실습에 없는 barplot
# seaborn barplot 함수 이용해서 'MultipleLines' 컬럼에 대해 bar 그려보자.
# x='MultipleLines', y='TotalCharges'

sns.barplot(data=df, x='MultipleLines', y='TotalCharges')



# tenure','MonthlyCharges','TotalCharges' 컬럼간의 상관관계를 seaborn heatmap으로 그려보자.(annot 추가)
# 분석결과 : tenure(서비스 사용기간)과 TotalCharges(서비스 총요금)간의 깊은 상관관계가 있어 보인다.

sns.heatmap(df[['tenure','MonthlyCharges','TotalCharges']].corr(), annot=True)



# seaborn boxplot 이용하기(x='Churn', y='TotalCharges')
# 분석결과
# - 이탈하는 고객이 이탈하지 않는 고객에 비해 총사용금액이 낮으며, Outlier 보인다.

sns.boxplot(data=df, x='Churn', y='TotalCharges')



# 판다스 to_csv 함수 이용해서 'data_v1_save.csv' 파일로 저장하기
# index=False 주어야 기존 인덱스 값이 저장되지 않음

df.to_csv('data_v1_save.csv', index=False)



# MultipleLines 컬럼의 값들이 문자열로 되어 있어 숫자로 변환해야 함. 컴퓨터가 이해할수 있도록
# Object 컬럼의 데이터를 원-핫-인코딩해서 숫자로 변경해 주는 함수 : Pandas get_dummies()

pd.get_dummies(data=df, columns=['MultipleLines'])



# PCA 임포트 및 PCA 수행
# PCA 정의시 주성분갯수 n_components=38 주어, 38개의 주성분을 만들게 하며
# 몇개의 주성분으로도 X데이터의 분산 80% 이상을 설명할수 있는지 확인해 보자.

from sklearn.decomposition import PCA

pca = PCA(n_components=38)
pca_components = pca.fit_transform( X )



# PCA의 explained_variance_ratio_ (설명된 분산 비율) 확인
# 첫번째 주성분만으로도 X 데이터의 99% 분산을 설명할수 있다!! (타겟변수가 심각한 불균형인지 확인 필요)

pca.explained_variance_ratio_



# Train dataset, Test dataset 나누기 : train_test_split 함수 사용
# 입력 : X, y
# Train : Test 비율 = 7: 3  --> test_size=0.3
# y Class 비율에 맞게 나누기 : stratify=y
# 여러번 수행해도 같은 결과 나오게 고정하기 : random_state=42
# 결과 : X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)



# 사이키런의 MinMaxScaler() 함수 활용
# 정의할 결과를 'scaler'로 매핑
# X_train과 X_test에 대해 MinMaxScaler 적용하기
# X_train에 대해서는 fit_transform 해주고, X_test에 대해서는 transform 해 주자

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# 모델별로 Recall 점수 저장
# 모델 Recall 점수 순서대로 바차트를 그려 모델별로 성능 확인 가능

from sklearn.metrics import accuracy_score

my_predictions = {}

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]

# 모델명, 예측값, 실제값을 주면 위의 plot_predictions 함수 호출하여 Scatter 그래프 그리며
# 모델별 MSE값을 Bar chart로 그려줌
def recall_eval(name_, pred, actual):
    global predictions
    global colors

    plt.figure(figsize=(12, 9))

    #acc = accuracy_score(actual, pred)
    acc = recall_score(actual, pred)
    my_predictions[name_] = acc * 100

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(y_value, columns=['model', 'recall'])
    print(df)

    length = len(df)

    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['recall'])

    for i, v in enumerate(df['recall']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    plt.title('recall', fontsize=18)
    plt.xlim(0, 100)

    plt.show()



# LogisticRegression 함수 사용 및 정의 : lg 저장
# 정의된 LogisticRegression 학습 fit() : 입력값으로 X_train, y_train 준다.
# 분류기 성능 평가(score)

lg = LogisticRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)

# 오차행렬
# TN  FP
# FN  TP

lg_pred = lg.predict(X_test)

confusion_matrix(y_test, lg_pred)
accuracy_score(y_test, lg_pred) # 정확도
precision_score(y_test, lg_pred) # 정밀도
recall_score(y_test, lg_pred) # 재현율 : 낮다.
f1_score(y_test, lg_pred) # 정밀도 + 재현율

print(classification_report(y_test, lg_pred))
recall_eval('LogisticRegression', lg_pred, y_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

recall_eval('K-Nearest Neighbor', knn_pred, y_test)

# DecisionTreeClassifier 학습 모델 : dt
# DecisionTreeClassifier 모델의 predict() 활용 : 입력값으로 X_test
# 결과 : dt_pred 저장

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

recall_eval('DecisionTree', dt_pred, y_test)

rfc = RandomForestClassifier(n_estimators=3, random_state=42)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

recall_eval('RandomForest Ensemble', rfc_pred, y_test)

# 램덤포레스트 모델의 변수중요도를 변수별로 중요도를 바챠트 출력
# Series 함수 이용 : data=rfc.feature_importances_ , index=X.columns
# sort_values 함수 활용하여 오름차순으로 정렬
# 바챠트 그리기

pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False).plot(kind='bar')

xgb = XGBClassifier(n_estimators=3, random_state=42)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

recall_eval('XGBoost', xgb_pred, y_test)

# 변수 중요도 그래프 그리기 : plot_importance

plot_importance()

lgbm = LGBMClassifier(n_estimators=3, random_state=42)

lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
recall_eval('LGBM', lgbm_pred, y_test)

lgbm.score(X_test, y_test)

recall_score(y_test, lgbm_pred)

lgbm.feature_importances_

pd.Series(lgbm.feature_importances_, index=X.columns).sort_values(ascending=False).plot(kind='bar')

# GridSearchCV 활용하여 xgb 모델의 최적의 하이퍼 파라미터를 찾습니다.
# 1. 최적의 파라미터를 찾을 학습기 XGB 정의 합니다.
# 2. 선정된 학습기에 대해 성능을 측정하고 싶은 하이퍼파라미터를 리스트를 정의 합니다.
# 3. GridSearchCV 함수는 선정된 학습기에 조합된 하이퍼파라미터를 하나씩 대입해서 학습 및 성능 측정합니다.
# 4. GridSearchCV 함수 수행이 완료 되면, 최적의 하이퍼파라미터을 찾을수 있게 됩니다.

from sklearn.model_selection import GridSearchCV

xgb = XGBClassifier(random_state=42)
param_xgb = {"max_depth": [10, 15], "n_estimators": [100,150] }

grid_xgb = GridSearchCV (estimator = xgb, param_grid = param_xgb, )
grid_xgb.fit(X_train, y_train)

grid_pred = grid_xgb.best_estimator_.predict(X_test)

recall_eval('Grid_XGB', grid_pred, y_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

tf.random.set_seed(100)

batch_size = 16
epochs = 20

# Sequential() 모델 정의 하고 model로 저장
# input layer는 input_shape=() 옵션을 사용한다.
# 38개 input layer
# unit 4개 hidden layer
# unit 3개 hidden layer
# 1개 output layser : 이진분류


model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(38,)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 앞쪽에서 정의된 모델 이름 : model
# Sequential 모델의 fit() 함수 사용
### X, y : X_train, y_train
### validation_data=(X_test, y_test)
### epochs 10번
### batch_size 10번


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',  metrics=['accuracy'])

# val_loss 모니터링해서 성능이 5번 지나도록 좋아지지 않으면 조기 종료, val_loss 가장 낮은 값을 가질때마다 모델저장

check_point = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_loss', save_best_only=True)

history = model.fit(x=X_train, y=y_train, epochs=50 , batch_size=20, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop, check_point])

pred = model.predict(X_test)
y_pred = np.argmax(pred, axis=1)

# SMOTE 함수 정의 및 Oversampling 수행

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)