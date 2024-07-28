filename ='Feature Website.xlsx'
#xlrd 지원불가로 openpyxl 사용할 수 있도록 변경
df = pd.read_excel(filename, engine='openpyxl')

def html_script_characters(soup):
    html_len = str(soup.script)
    return float(len(html_len.replace(' ', '')))

script_len = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    script_len.append(html_script_characters(soup))

df['html_script_characters'] = script_len

def html_num_whitespace(soup):
    try:
        # soup > body > text > count
        NullCount = soup.body.text.count(' ')
        return float(NullCount)
    except:
        return 0.0

num_whitespace = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    num_whitespace.append(html_num_whitespace(soup))

df['html_num_whitespace'] = num_whitespace

def html_num_characters(soup):
    try:
        #soup > body > text
        bodyLen = len(soup.body.text)
        return float(bodyLen)
    except:
        return 0.0

html_body = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    html_body.append(html_num_characters(soup))

df['html_body_len'] = html_body

def html_link_in_script(soup):
    numOfLinks = len(soup.findAll('script', {"src": True}))
    numOfLinks += len(soup.findAll('script', {"href": True}))
    return float(numOfLinks)

html_script_link_num = []

for index, row in df.iterrows():
    soup = BeautifulSoup(row.html_code, 'html.parser')
    html_script_link_num.append(html_link_in_script(soup))

df['html_script_link_num'] = html_script_link_num



df = pd.read_csv('TrainData.csv',delimiter=',')

df=df.drop_duplicates()

import pandas as pd

df_ex = pd.DataFrame({'name': ['Alice','Bob','Charlie','Dave','Ellen','Frank'], 'age': [24,42,18,68,24,30],  'state': ['NY','CA','CA','TX','CA','NY'], 'point': [64,24,70,70,88,57]})

df_ex['state'].replace({'CA':'California','NY':'NewYork'}, inplace=True)
print(df_ex)

# unique() 유일한 값 확인
df['Result_v1'].unique()

# replace() 함수 사용
df['Result_v1'].replace({'benign':1,'malicious':-1}, inplace=True)

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df_ex = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_ex['target'] = iris.target

print(iris.DESCR)
df_ex.corr()['target'].sort_values(ascending=False)

df_ex['color'] = df_ex['target'].map({0:"red", 1:"blue", 2:"green"})

df.drop(columns=["url_chinese_present","html_num_tags('applet')"],inplace=True)

X = df.iloc[:,0:len(df.columns)-1].values
y = df.iloc[:,len(df.columns)-1].values

train_x, val_x, train_y, val_y = train_test_split(X, y,test_size=0.3,random_state=2021)



def plot_confusion_matrix(ax, matrix, labels = ['malicious','benign'], title='Confusion matrix', fontsize=9):
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])

    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation='90', fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], fontsize=fontsize, minor=True)

    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')

    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)

    # Plot heat map
    proportions = [1. * row / sum(row) for row in matrix]
    ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Blues)

    # Plot counts as text
    for row in range(len(matrix)):
        for col in range(len(matrix[row])):
            confusion = matrix[::-1][row][col]
            if confusion != 0:
                ax.text(col + 0.5, row + 0.5, int(confusion),
                        fontsize=fontsize,
                        horizontalalignment='center',
                        verticalalignment='center')

    # Add finishing touches
    ax.grid(True, linestyle=':')
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel('prediction', fontsize=fontsize)
    ax.set_ylabel('actual', fontsize=fontsize)

    plt.show()

param_grid = [ {'n_estimators':[30,40,50,60,100], 'max_depth':[30,40,50,60,100]}]

rfc = RandomForestClassifier()

# Classification일때  'accuracy','f1' # Regression 일때 'neg_mean_squared_error','r2'... # Log 출력 Level 조정
rfc_grid = GridSearchCV(rfc, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=1)

rfc_grid.fit(train_x, train_y)

rfc_model = rfc_grid.best_estimator_

print('최적의 파라미터 값 : ', rfc_grid.best_params_)
print('최고의 점수 : ', rfc_grid.best_score_)

rfc_grid_pred = rfc_model.predict(val_x)

# train 및 val 데이터 정확도
rfc_model.score(train_x, train_y), rfc_model.score(val_x, val_y)



# - 리스트.append(요소) : 리스트의 끝에 요소 추가
# - 리스트.insert(인덱스, 요소) : 리스트의 특정 인덱스에 요소 추가
# - 리스트1.extend(리스트2) : 리스트1에 리스트2를 연결해 확장
# - 리스트1 + 리스트2 : 리스트1과 리스트2를 서로 병합
# - 리스트.remove(요소) : 리스트의 특정 값의 요소를 삭제
# - 리스트.count('요소') : 리스트 중 특정 값을 가진 요소의 개수를 카운트
# - 리스트.sort() : 리스트 내 요소를 오름차순으로 정렬
# - 리스트.pop(인덱스) : 리스트 중 특정 인덱스의 요소를 삭제



cust2 = pd.read_csv('./sc_cust_info_txn_v1.5.csv', index_col='cust_class', usecols=['cust_class', 'r3m_avg_bill_amt', 'r3m_B_avg_arpu_amt', 'r6m_B_avg_arpu_amt'], encoding = "cp949")

cust.cust_class = cust['cust_class'] # cf : series 형태로 가지고 오기(hcust.cust_class = cust['cust_class'])
cust.cust_class = cust[['cust_class']] # cf : Dataframe형태로 가지고 오기

cust.loc[[102, 202, 302]] #여러개의 row 가지고 오기
cust.iloc[[2, 102, 202]] #iloc과비교(위와 같은 값을 가지고 오려면...)

gender_group = cust.groupby('sex_type')
gender_group.groups

# reset_index활용하여 기존 DataFrame으로 변환  (set_index <-> reset_index)

cust.set_index(['cust_class','sex_type']).reset_index()

# 멀티 인덱스 셋팅 후 인덱스 기준으로 groupby하기
# 'sex'와 'cp'를 기준으로 index를셋팅하고 index를 기준으로 groupby하고자 하는경우
# groupby의 level은 index가 있는 경우에 사용

cust.set_index(['cust_class','sex_type']).groupby(level=[0,1]).mean()

#그룹별로 한번에 데이터를 한번에 보는 경우

cust.set_index(['cust_class','sex_type']).groupby(level=[0,1]).aggregate([np.mean, np.max])

# 행(row)는 고객ID(cust_id), 열(col)은 상품코드(prod_cd), 값은 구매금액(pch_amt)을 pivot릏 활용하여 만들어보기

data.pivot(index = 'cust_id', columns ='prod_cd', values ='pch_amt')
data.pivot_table(index = ['cust_id','grade'], columns ='prod_cd', values ='pch_amt')

data.pivot(index = 'cust_id', columns =['grade','prod_cd'], values ='pch_amt')
data.pivot_table(index = 'cust_id', columns =['grade','prod_cd'], values ='pch_amt')

data.pivot_table(index='grade', columns='prod_cd', values='pch_amt')

data.pivot_table(index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.sum)
data.pivot_table(index='grade', columns='prod_cd', values='pch_amt', aggfunc=np.mean)

new_df = df.set_index(['지역', '요일'])
new_df.unstack(0).stack(0)

df1 = pd.DataFrame({'key1' : [0,1,2,3,4], 'value1' : ['a', 'b', 'c','d','e']}, index=[0,1,2,3,4])
df2 = pd.DataFrame({'key1' : [3,4,5,6,7], 'value1' : ['c','d','e','f','g']}, index=[3,4,5,6,7])

pd.concat([df1, df2], ignore_index=False)
pd.concat([df1, df2], axis =1)

pd.concat([df1, df2], join='outer')
pd.concat([df1, df2], join='inner')

pd.concat([df1, df2], verify_integrity=False)

pd.merge(cust1, order1, left_index=True, right_index=True)

pd.merge(customer, orders, on='cust_id', how='right').groupby('item').sum().sort_values(by='quantity', ascending=False)

pd.merge(customer, orders, on='cust_id', how='inner').groupby(['name', 'item']).sum().loc['영희']



cust=cust.astype({'age': int})

# 파이썬에서 Copy 메소드를 사용하지 않으면 주소값을 복사해서 사용기 때문에 원본 값을 변경 시키게 됩니다.
# 따라서 원본 항목을 보전하면서 데이터를 보정하려면 copy 메소드를 사용 해주셔야 합니다.

# 뒤에 있는 data를 사용해서 결측치를 처리하는 방법
cust=cust.fillna(method='backfill')

# 앞에 있는 data를 사용해서 결측치를 처리하는 방법
cust=cust.fillna(method='ffill')

#replace()함수로 결측치 채우기
cust['age']=cust['age'].replace(np.nan, cust['age'].median())

# interpolate 함수의 선형 방법을 사용하여 결측값을 채우기
cust=cust.interpolate()

#listwise 방식으로 제거 하기
cust=cust.dropna()

#pairwise 방식으로 제거하기
cust=cust.dropna(how='all')

#임계치를 설정해서 제거하기
cust=cust.dropna(thresh=10)

# 특정열 안에서만 삭제하기|
cust=cust.dropna(subset=['class'])

print(cust['sex'].value_counts())

cust_data=cust[(cust['class']!='H')]

cust_data['class']=cust_data['class'].replace('H','F')

#이상치를 제거하는 함수 만들기
def removeOutliers(x, column):
    # Q1, Q3구하기
    q1 = x[column].quantile(0.25)
    q3 = x[column].quantile(0.75)

    # 1.5 * IQR(Q3 - Q1)
    iqr = 1.5 * (q3 - q1)

    # 이상치를 제거
    y=x[(x[column] < (q3 + iqr)) & (x[column] > (q1 - iqr))]

    return(y)

cust_data=removeOutliers(cust, 'avg_bill')

# 이상치를 변경하는 함수 만들기
def changeOutliers(data, column):
    x=data.copy()
    # Q1, Q3구하기
    q1 = x[column].quantile(0.25)
    q3 = x[column].quantile(0.75)

    # 1.5 * IQR(Q3 - Q1)
    iqr = 1.5 * (q3 - q1)

    #이상치 대체값 설정하기
    Min = 0
    if (q1 - iqr) > 0 : Min=(q1 - iqr)

    Max = q3 + iqr

    # 이상치를 변경
    # X의 값을 직졉 변경해도 되지만 Pyhon Warning이 나오기 떄문에 인덱스를 이용
    x.loc[(x[column] > Max), column]= Max
    x.loc[(x[column] < Min), column]= Min

    return(x)

cust_data=changeOutliers(cust, 'avg_bill')

# age를 활용하여 나이대("by_age")  Feature 만들기
cust_data['by_age']=cust_data['age']//10*10
cust_data=cust_data.astype({'age': int, 'by_age':int})

q1=cust_data['avg_bill'].quantile(0.25)

#cut 메소드를 활용하여 요금을 3개 구간으로 나누기
cust_data['bill_rating'] = pd.cut(cust_data["avg_bill"], bins=[0,q1,q3,cust_data["avg_bill"].max()] , labels=['low', 'mid','high'])

#qcut 메소드를 활용하여 요금을 동일 비율로 3개 구간으로 나누기
cust_data['bill_rating'] = pd.qcut(cust_data["avg_bill"], 3 , labels=['low', 'mid','high'])

# 표준화
Standardization_df = (cust_data_num - cust_data_num.mean())/cust_data_num.std()

#사이킷런 패키지의 MinMaxScaler를 이용하여  Scaling 하기

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
normalization_df=cust_data_num.copy()
normalization_df[:]=scaler.fit_transform(normalization_df[:])

#MinMaxScaler 함수 구현하기
normalization_df = (cust_data_num - cust_data_num.min())/(cust_data_num.max()-cust_data_num.min())

#pandas에서는 get_dummies함수를 사용하면 쉽게 One-Hot Encording이 가능
pd.get_dummies(cust_data['class'])

#columns를 사용해서 기존 테이블에 One-Hot Encording으로 치환된 변수 생성하기
cust_data_end=pd.get_dummies(cust_data, columns=['class'])



plt.figure(figsize=(16,5))
plt.plot([1,2,3], [100,120,110])
plt.show()

plt.scatter(y=df["avg_bill"], x=df["age"])

plt.hist(df["age"],bins=20) #기본 bin은 10개

df.boxplot(by="by_age", column="avg_bill", figsize=(16,8)) # by는 Group화 할 값(컬럼), column은 박스 그래프로 나타낼 값(컬럼)

plt.bar(x, y)

pd.pivot_table(df, index = ['service'])

df2[['A_bill', 'B_bill']].plot(kind='bar', stacked=True) #stacked 누적 여부

plt.plot(x,y1,'r--', x,y2, 'bs' ,x, y3, 'g^:')

# plt.subplot(221)는 2행 2열 배치에 첫번째 영역이 됩니다.

sns.scatterplot(x='age', y='avg_bill', data=df)

sns.catplot(x='age', y='avg_bill',data=df ,col="class", col_wrap=2)

sns.lmplot(x='avg_bill', y='B_bill', data=df,line_kws={'color': 'red'})

sns.lmplot(x='avg_bill', y='B_bill', data=df, hue='sex')

sns.histplot(x="age",bins=20, hue="bill_rating",data=df, multiple='dodge', shrink=0.8)
sns.countplot(y="class", hue="sex", data=df)

sns.countplot(y="class", hue="sex", data=df, palette='spring')

sns.jointplot(x="avg_bill", y="age", data=df, kind='hex')

sns.heatmap(df.corr()) # 컬럼별 상관관계

sns.boxplot(y=df["avg_bill"], x=df["by_age"],width=0.9)

sns.violinplot(y=df["A_bill"], x=df["class"],width=1)



X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델별로 Accuracy 점수 저장
# 모델 Accuracy 점수 순서대로 바차트를 그려 모델별로 성능 확인 가능

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
def accuracy_eval(name_, pred, actual):
    global predictions
    global colors

    plt.figure(figsize=(12, 9))

    acc = accuracy_score(actual, pred)
    my_predictions[name_] = acc * 100

    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=True)

    df = pd.DataFrame(y_value, columns=['model', 'accuracy'])
    print(df)

    length = len(df)

    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['accuracy'])

    for i, v in enumerate(df['accuracy']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v + 2, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    plt.title('accuracy', fontsize=18)
    plt.xlim(0, 100)

    plt.show()

final_outputs = {'DecisionTree': dt_pred, 'randomforest': rfc_pred, 'xgb': xgb_pred, 'lgbm': lgbm_pred,'stacking': stacking_pred,}

final_prediction= final_outputs['DecisionTree'] * 0.1+final_outputs['randomforest'] * 0.2+final_outputs['xgb'] * 0.25+final_outputs['lgbm'] * 0.15+final_outputs['stacking'] * 0.3\

# 가중치 계산값이 0.5 초과하면 1, 그렇지 않으면 0
final_prediction = np.where(final_prediction > 0.5, 1, 0)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 모델 컴파일 – 다중 분류 모델 (Y값을 One-Hot-Encoding 한경우)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 컴파일 – 다중 분류 모델 (Y값을 One-Hot-Encoding 하지 않은 경우)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback : 조기종료, 모델 저장

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
check_point = ModelCheckpoint('best_model.h5', verbose=1, monitor='val_loss', mode='min', save_best_only=True)

history = model.fit(x=X_train, y=y_train, epochs=50 , batch_size=20,validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop, check_point])

losses = pd.DataFrame(model.history.history)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 18개 input layer
# unit 4개 hidden layer
# unit 3개 hidden layer
# 1개 output layser : 이진분류

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(18,)))
model.add(Dropout(0.3))
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

losses[['loss','val_loss', 'accuracy','val_accuracy']].plot()

# dataset-clean_dirty.zip 파일을 IMAGE 디렉토리로 복사 및 압축 풀기
if not os.path.exists('IMAGE') :
    !mkdir IMAGE
    !cp dataset-clean_dirty.zip ./IMAGE
    !cd IMAGE ; unzip dataset-clean_dirty.zip

# ./IMAGE/clean 폴더 안의 이미지 갯수
!ls -l ./IMAGE/clean | grep jpg | wc -l

clean_img_path = './IMAGE/clean/plastic1.jpg'

gfile = tf.io.read_file(clean_img_path)
image = tf.io.decode_image(gfile, dtype=tf.float32)

# Hyperparameter Tunning

num_epochs = 50
batch_size = 4
learning_rate = 0.001

input_shape = (384, 512, 3)  # 사이즈 확인
num_classes = 2    # clean, dirty

# ImageDataGenerator 이용하여 이미지 전처리하기

training_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)  # train set : 435 * (1 - 0.2) = 348
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # test set : 435 * 0.2 = 87

# 이미지 데이터 읽고 배치 , 셔플하고 labeling 수행

# IMAGE 포더 밑에 .ipynb_checkpoints 폴더 있을경우 폴데 삭제
!rm -rf ./IMAGE/.ipynb_checkpoints

training_generator = training_datagen.flow_from_directory('./IMAGE/', batch_size=batch_size, target_size=(384, 512), class_mode = 'categorical',shuffle = True, subset = 'training')
test_generator = training_datagen.flow_from_directory('./IMAGE/', batch_size=batch_size, target_size=(384, 512), class_mode = 'categorical',shuffle = True, subset = 'validation')

# class 이름 및 번호 매핑 확인
print(training_generator.class_indices)

batch_samples = next(iter(training_generator))

print('True Value : ',batch_samples[1][0])
plt.imshow(batch_samples[0][0])
plt.show()

# Feature extraction
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Classification
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(training_generator, epochs=3 ,steps_per_epoch = len(training_generator) / batch_size,validation_steps = len(test_generator) / batch_size,validation_data=test_generator, verbose=1)

# test_generator 샘플 데이터 가져오기
# 배치 사이즈 32 확인

batch_img, batch_label = next(iter(test_generator))
print(batch_img.shape)
print(batch_label.shape)

# 4개 Test 샘플 이미지 그려보고 예측해 보기

i = 1
plt.figure(figsize=(16, 30))
for img, label in list(zip(batch_img, batch_label)):
    pred = model.predict(img.reshape(-1,384, 512,3))
    pred_t = np.argmax(pred)
    plt.subplot(8, 4, i)
    plt.title(f'True Value:{np.argmax(label)}, Pred Value: {pred_t}')
    plt.imshow(img)
    i = i + 1

X_train = X_train.reshape(-1,18,1)
X_test = X_test.reshape(-1,18,1)

# define model
model = Sequential()
model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(18, 1)))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))



#암호화를 위한 함수를 정의합니다.
def encrypt(target):
    hashSHA=hashlib.sha256() #SHA256 해시 객체 생성
    hashSHA.update(str(target).encode('utf-8')) #해시 값 생성 암호화 대상이 숫자형인 경우에는 문자로 변경해야 한다.
    return hashSHA.hexdigest().upper() #해시값 반환

ansan_data['IS']=ansan_data['IS'].apply(encrypt)

with open('json_data.pickle', 'rb') as f:
    json_data = pickle.load(f) # 단 한줄씩 읽어옴

#실습 코드 - for문과 if문을 활용하여 도로명 주소와 주소에 대한 좌표를 읽어서 address_data에 저장하세요
for i in np.arange(len(json_data)):
    if json_data[i].json()['documents']!=[]: #주소가 조회되지 않은 경우가 존재하므로 제외함
        address_data=address_data.append([(json_data[i].json()['documents'][0]['road_address']['address_name'],
                                           json_data[i].json()['documents'][0]['road_address']['x'],
                                           json_data[i].json()['documents'][0]['road_address']['y'])],
                                         ignore_index=True)

#실습 코드 - 데이터프레임의 중복을 제거합니다.(dop_duplicates()를 사용하세요)
ansan_data = ansan_data.drop_duplicates()

#실습 코드 - 상세분류2 --> TYPE_DTL_NM로 컬럼명을 변경합니다. rename 함수를 이용하세요. 입력값은 {변경전컬럼명,변경후컬럼명} 이며, axis=1 옵션을 주어야 합니다.
ansan_data = ansan_data.rename({'상세분류2':'TYPE_DTL_NM'},axis=1)

#실습 코드 PRODUCT가 '059Z'인 데이터를 삭제합니다.
ansan_data=ansan_data.loc[(ansan_data['PRODUCT']!='059Z')]

#실습 코드 - TYPE과 TYPE_DTL을 .groupby() 함수를 이용해서 살펴봅니다. > count() 사용
pd.DataFrame(ansan_data.groupby(['TYPE','TYPE_DTL_NM'])['PRODUCT'].count())

#실습 코드 - to_csv 함수를 이용하여 ansan_data 를 CSV파일로 저장합니다. 경로명(aidu환경): sacp_framework.config.data_dir + '/ansan_data.csv'
ansan_data.to_csv(sacp_framework.config.data_dir + '/ansan_data_pre.csv',encoding='utf-8',header=True, index=False)

#대상이 되는 Dataframe과 기타상품을 구분하는 임계값을 입력으로 받습니다.
def etctoetc(df,threshold):
    total_cnt = df['CNT'].sum()
    cnt=0
    for n in df.index:
        cnt=cnt+df.loc[n,'CNT'] #루프를 돌때마다 cnt의 값을 PRODUCT만큼 증가시킴
        if(cnt > total_cnt*threshold): #cnt의 값이 total_cnt*threshold보다 크면
            df.loc[n:,'PRODUCT_NM'] = '기타상품' #n번 인덱스 이후의 PRODUCT_NM을 '기타상품'으로 상품명을 변경
            break
    #기타상품으로 변경된 상품을 하나로 합쳐줌
    df = pd.DataFrame(df.groupby(['TYPE','PRODUCT_NM'])['CNT'].sum()).reset_index()
    #데이터를 정렬할 때 기타상품이 가장 뒤에 오도록 수정
    df2 = df.loc[df['PRODUCT_NM']=='기타상품',:]
    df3 = df.loc[df['PRODUCT_NM']!='기타상품',:].sort_values(by=['CNT'], ascending=False)
    df4= pd.concat([df3,df2],sort=False).reset_index(drop=True)

    return df4

#실습 코드 - x축은 PRODUCT_NM, y축은 CNT가 되도록 개인, 개인사업자, 법인에 대하여 barplot 그래프를 그리세요.
i=1
for c_type in ['개인','개인사업자','법인']:
    plt.subplot(3,1,i)
    graph_data=etctoetc(prod_stat[prod_stat['TYPE']==c_type],0.9)
    sns.barplot(data = graph_data,x='PRODUCT_NM',y='CNT')
    i=i+1

plt.show()

import folium

map = folium.Map(location=[f_lat,f_lon], zoom_start=14)
map_ST = folium.Map(location=[f_lat,f_lon], zoom_start=14, tiles='Stamen Terrain') # Mapbox 의 경우에는 API key가 필요

# Parameter
# data (list of points of the form [lat, lng] or [lat, lng, weight]) - numpy.array 또는 list로 입력
# min_opacity (default 1.) – 불투명도
# max_zoom (default 18) – 최대 밀도에 도달하는 Zoom 레벨
# radius (int, default 25) – 포인트의 반경

#Train set의 크기 계산
N=int(len(ansan_data_T) * 0.9)

#sample 함수를 이용해 Train_set을 분리하여 저장
train_data=ansan_data_T.sample(n=N)

#Train_set을 제외한 나머지를 Test_set으로 저장
test_data=ansan_data_T.drop(ansan_data_T.index[train_data.index])

le_columns=train_data.columns

from sklearn.preprocessing import LabelEncoder

# #실습 코드 - LabelEncoder를 객체로 생성한 후 , fit( ) 과 transform( ) 으로 label 인코딩 수행.
le = LabelEncoder()

for column in le_columns:
    le.fit(train_data[column])
    train_data[column]=le.transform(train_data[column])
    #train_data에 없는 label이 test_data에 있을 수 있으므로 아래 코드가 필요하며, test_data는 fit 없이 transform만 해야함
    for label in np.unique(test_data[column]):
        if label not in le.classes_: # unseen label 데이터인 경우( )
            le.classes_ = np.append(le.classes_, label) # 미처리 시 ValueError발생
    test_data[column]=le.transform(test_data[column])

# test_size: test_size의 크기
# random_state: random_state를 동일하게 유지해야 일정한 데이터가 나옴
# stratify: label을 균등하게 분포하도록 함

#실습 코드 사이킷런의 train_test_split로 데이터를 train셋과 validation셋으로 분리하세요.
from sklearn.model_selection import train_test_split

x_train_le, x_val_le, y_train_le, y_val_le = train_test_split(train_data_le.drop('TYPE',1),train_data_le['TYPE'],test_size=0.2,random_state=21,stratify=train_data_le['TYPE'])

x_train_le.shape, x_val_le.shape, y_train_le.shape, y_val_le.shape

import matplotlib.pyplot as plt
import seaborn as sns

my_predictions = {}

colors = ['r', 'c', 'm', 'y', 'k', 'khaki', 'teal', 'orchid', 'sandybrown',
          'greenyellow', 'dodgerblue', 'deepskyblue', 'rosybrown', 'firebrick',
          'deeppink', 'crimson', 'salmon', 'darkred', 'olivedrab', 'olive',
          'forestgreen', 'royalblue', 'indigo', 'navy', 'mediumpurple', 'chocolate',
          'gold', 'darkorange', 'seagreen', 'turquoise', 'steelblue', 'slategray',
          'peru', 'midnightblue', 'slateblue', 'dimgray', 'cadetblue', 'tomato'
         ]


#acc를 구해서 시각화해줌
def acc_eval(name_, pred, actual):
    global predictions
    global colors

    acc = (pred==actual).mean()
    my_predictions[name_] = acc


    y_value = sorted(my_predictions.items(), key=lambda x: x[1], reverse=False)  # 정확도 내림차순으로 sort

    df = pd.DataFrame(y_value, columns=['model', 'acc'])
    #print(df)
    min_ = df['acc'].min() -0.5
    max_ = 1.2

    length = len(df)

    plt.figure(figsize=(10, length))
    ax = plt.subplot()
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df['model'], fontsize=15)
    bars = ax.barh(np.arange(len(df)), df['acc'])

    for i, v in enumerate(df['acc']):
        idx = np.random.choice(len(colors))
        bars[i].set_color(colors[idx])
        ax.text(v+0.1, i, str(round(v, 3)), color='k', fontsize=15, fontweight='bold')

    plt.title('Accuracy', fontsize=18)
    plt.xlim(min_,max_)

    plt.show()

#실수로 잘못 넣은 경우 해당 모델을 삭제
def remove_model(name_):
    global my_predictions
    try:
        del my_predictions[name_]
    except KeyError:
        return False
    return True

#실습 코드 - 모델을 생성합니다. 파라미터 수정
model_dt = DecisionTreeClassifier(min_samples_split=2,
                                  min_samples_leaf=1,
                                  max_features=None,
                                  max_depth=None,
                                  max_leaf_nodes=None
                                 )

n_nodes = model_dt.tree_.node_count
children_left = model_dt.tree_.children_left
children_right = model_dt.tree_.children_right
feature = model_dt.tree_.feature
threshold = model_dt.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has {n} nodes and has "
      "the following tree structure:\n".format(n=n_nodes))
for i in range(n_nodes):
    if is_leaves[i]:
        print("{space}node={node} is a leaf node.".format(
            space=node_depth[i] * "\t", node=i))
    else:
        print("{space}node={node} is a split node: "
              "go to node {left} if X[:, {feature}] <= {threshold} "
              "else to node {right}.".format(
                  space=node_depth[i] * "\t",
                  node=i,
                  left=children_left[i],
                  feature=feature[i],
                  threshold=threshold[i],
                  right=children_right[i]))



model_rf= RandomForestClassifier(n_jobs=-1,n_estimators=300, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_depth=None, max_leaf_nodes=None)

# #### **1. Tree Boost Parameter**
# - **eta**: learning rate
# - **num_boost_around**: 생성할 weak learner의 수
# - **min_child_weight**: 관측치에 대한 가중치 합의 최소로 과적합 조절 용도 (값이 크면 과적합이 감소)
# - **gamma**:리프 노드에서 추가분할을 만드는데 필요한 최소 손실감소 값(값이 크면 과적합이 감소)
# - **max_depth**:Tree 의 최대 깊이(너무 크면 과적합)
# - **sub_sample**:훈련 중 데이터 샘플링 비율 지정(과적합 제어)
# - **colsample_bytree**: 열의 서브 샘플링 비율

# **2. Learning Task Parameter**
# - **objective**
#         reg:linear : 회귀
#         binary:logistic : 이진분류
#         multi:softmax : 다중분류, 클래스 반환
#         multi:softprob : 다중분류, 확률 반환
# - **eval_metric**
#         rmse : Root Mean Squared Error<br>
#         mae : mean absolute error<br>
#         logloss : Negative log-likelihood<br>
#         error : binary classification error rate<br>
#         merror : multiclass classification error rate<br>
#         mlogloss: Multiclass logloss<br>
#         auc: Area Under Curve<br>

# xgb 학습시 train 데이터 세트는 'train', validation 데이터 세트는 'eval' 로 명기
wlist = [(dtrain, 'train'), (dval,'eval')]

#실습 코드 - output_margin=True 옵션을 넣어 예측합니다.

pred_xgb_margin=model_xgb.predict(dtest, output_margin=True)


# ROC Curve
# 단 y_true , y_pred 는 n_classes 갯수 만큼
#One-Hot 인코딩 된 상태의 2 차원 이상으로 구성되어야 함
sacp_framework.plot_roc_curve(
    # Label(Y) 의 정답 (numpy.array)
    y_true=pd.get_dummies(y_true).to_numpy(),
    # Label(Y) 의 예측결과 (numpy.array)
    y_pred=pred_xgb_margin,
    n_classes=len(y_label), # Label(Y) 의 이름 갯수
    target_names=y_label, # Label(Y) 의 이름 (array)
    title='ROC Curve' # matplotlib 객체의 타이틀 (옵션)
)



from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm
print("--에러 메시지가 나타나지 않았다면 정상적으로 import된 것입니다.--")

aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)
data_jan = pd.read_csv("air_data_01.csv", header=0, encoding='cp949')

data_jan_dropped = data_jan.dropna(how='all', axis='columns')

# Hint : 배열을 사용해 데이터를 불러오면 반복적으로 호출하기 쉽습니다.
# result = pd.DataFrame()

coloumns_we_need = ['3', '6', '8', '15']
dust = '초미세먼지'

result = pd.DataFrame()

for i in range(1, 12):
    print("data loading..."+str(i))
    data = pd.read_csv("air_data_"+str(i).zfill(2)+".csv", header=0, encoding='cp949')
    data_dropped = data[coloumns_we_need]
    cond_dust = data_dropped['6'] == dust

    data_dust = data_dropped[cond_dust]

    data_dust_day = data_dust.groupby(['15', '3']).median()
    data_dust_day = data_dust_day.reset_index()

    new_df = pd.DataFrame()
    new_df['date'] = data_dust_day['15']
    new_df['dev'] = data_dust_day['3']
    new_df['val'] = data_dust_day['8']

    result = pd.concat([result, new_df])

result.info()


for item in result['dev'].unique():
    cond_result = result['dev']==item
    temp = result[cond_result]
    temp.to_csv("data_"+str(item)+".csv",index=False)

# Hint : 가장 간단한 모델의 형태는 Prophet(yearly_seasonality=True)일 것입니다.
model = Prophet(yearly_seasonality=True)

# Hint : ARIMA(학습할 데이터, (p, d, q))로 모델을 생성할 수 있습니다.
# Hint : 모델의 학습은 fit()을 사용합니다.
model_new = ARIMA(train,(p,d,q))
model_fit = model_new.fit()

# Hint : 학습된 결과에 forecast()를 사용합니다. 이 떄 인수로 steps에 일 수를 지정해줍니다.
full_forecast = model_fit.forecast(steps=test.shape[0])
forecast = pd.DataFrame(full_forecast[0], index=test.index, columns = test.columns)



from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm
import numpy as np
import pandas as pd
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)
data = pd.read_csv("data_2.csv")

# Hint : 앞서 진행했던 내용을 참고하시면 됩니다. 특히 칼럼명을 변형하는 과정까지 진행해야 함을 기억하세요.
from fbprophet import Prophet
import datetime
data_dropped = data.drop('dev', axis='columns')
data_dropped['date'] = pd.to_datetime(data_dropped['date'],format='%Y%m%d')
data_dropped = data_dropped.set_index('date')
train_fb = data_dropped.loc[:datetime.datetime(2019,10,31),:]
test_fb = data_dropped.loc[datetime.datetime(2019,11,1):,:]

train_fb = train_fb.reset_index().rename(columns={'val':'y'})
train_fb['ds'] = pd.to_datetime(train_fb['date'].astype(str))
train_fb = train_fb.drop('date',axis=1)

test_fb = test_fb.reset_index().rename(columns={'val':'y'})
test_fb['ds'] = pd.to_datetime(test_fb['date'].astype(str))
test_fb = test_fb.drop('date',axis=1)

# Hint : 앞서 진행했던 내용을 참고하시면 됩니다.
from statsmodels.tsa.arima_model import ARIMA
import datetime
data_dropped = data.drop('dev', axis='columns')
data_dropped['date'] = pd.to_datetime(data_dropped['date'],format='%Y%m%d')
data_dropped = data_dropped.set_index('date')
train_arima = data_dropped.loc[:datetime.datetime(2019,10,31),:]
test_arima = data_dropped.loc[datetime.datetime(2019,11,1):,:]

# Commented out IPython magic to ensure Python compatibility.
# fbprophet

## 트렌드가 변화하는 지점을 더 잘 찾기 위한 특성
## range 기본 값은 0.8, scale 기본은 0.05
model_1 = Prophet(yearly_seasonality=True, changepoint_range=0.9, changepoint_prior_scale=0.1)
model_1.fit(train_fb)

forecast_fb_1 = model_1.predict(test_fb)

import matplotlib.pyplot as plt
# %matplotlib inline
fig,ax=plt.subplots(figsize=(16,13))
model_1.plot(forecast_fb_1, ax=ax)
plt.plot(train_fb['ds'],train_fb['y'])
plt.plot(test_fb['ds'],test_fb['y'])

## 특성이 없는 모델
model_2 = Prophet()
model_2.fit(train_fb)
forecast_fb_2 = model_2.predict(test_fb)
fig,ax=plt.subplots(figsize=(16,13))
model_2.plot(forecast_fb_2, ax=ax)
plt.plot(train_fb['ds'],train_fb['y'])
plt.plot(test_fb['ds'],test_fb['y'])

## 특성이 더 강한 모델
model_3 = Prophet(changepoint_prior_scale=0.5)
model_3.fit(train_fb)
forecast_fb_3 = model_3.predict(test_fb)
fig,ax=plt.subplots(figsize=(16,13))
model_3.plot(forecast_fb_3, ax=ax)
plt.plot(train_fb['ds'],train_fb['y'])
plt.plot(test_fb['ds'],test_fb['y'])

# ARIMA 모델
## ACF와 PACF 계산

import matplotlib.pyplot as plt
import statsmodels.api as sm
fig = plt.figure(figsize=(16,10))
# 그래프의 위치 설정
ax1=fig.add_subplot(211)
# ACF 그래프
fig = sm.graphics.tsa.plot_acf(train_arima['val'], lags=20, ax=ax1)
# 그래프의 위치 설정
ax2 = fig.add_subplot(212)
# PCAF 그래프
fig = sm.graphics.tsa.plot_pacf(train_arima['val'], lags=20,ax=ax2)

import copy as cp
differed = cp.deepcopy(train_arima)
differed = differed.diff().dropna()

fig = plt.figure(figsize=(16,10))
# 그래프의 위치 설정
ax1=fig.add_subplot(211)
# ACF 그래프
fig = sm.graphics.tsa.plot_acf(differed['val'], lags=20, ax=ax1)
# 그래프의 위치 설정
ax2 = fig.add_subplot(212)
# PCAF 그래프
fig = sm.graphics.tsa.plot_pacf(differed['val'], lags=20,ax=ax2)

p = 3
d = 0
q = 3

model_new = ARIMA(train_arima,(p,d,q))
model_fit = model_new.fit(trend='nc')
print(model_fit.summary())
full_forecast = model_fit.forecast(steps=train_arima.shape[0])
forecast_arima = pd.DataFrame(full_forecast[0], index=train_arima.index, columns = test_arima.columns)
plt.figure(figsize=(20,10))
plt.plot(train_arima)
plt.plot(test_arima)
plt.plot(forecast_arima)

p = 1
d = 1
q = 1

model_new = ARIMA(train_arima,(p,d,q))
model_fit = model_new.fit(trend='nc')
full_forecast = model_fit.forecast(steps=test_arima.shape[0])
forecast_arima = pd.DataFrame(full_forecast[0], index=test_arima.index, columns = test_arima.columns)
plt.figure(figsize=(20,10))
plt.plot(train_arima)
plt.plot(test_arima)
plt.plot(forecast_arima)



##### 3) ScreePlot을 활용한 요인수 결정 : Elbow 기법
* 고유값(각각의 요인으로 설명할 수 있는 변수들의 분산 총합) 시각화

# 요인분석 오브젝트를 만들고 실행해보겠습니다.
fa = FactorAnalyzer()
fa.set_params(rotation=None)
fa.fit(df.drop(['RID','TIME_DEPARTUREDATE','TIME_ARRIVEDATE'],axis=1))
# 고유값 확인 * 고유값(eigenvalue):각각의 요인으로 설명할 수 있는 변수들의 분산 총합
ev, v = fa.get_eigenvalues()
ev

# 요인분석 오브젝트를 만들고 실행해보겠습니다.
fa = FactorAnalyzer()
fa.set_params(n_factors=3, rotation=None)
fa.fit(df.drop(['RID','TIME_DEPARTUREDATE','TIME_ARRIVEDATE'],axis=1))
pd.DataFrame(fa.loadings_) # 요인부하량 확인 : 0.4이상 유의미, 0.5이상 중요

# 크론바흐 계수를 계산하는 함수를 선언하겠습니다.
def CronbachAlpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=0, ddof=1)
    tscores = itemscores.sum(axis=1)
    nitems = itemscores.shape[1]
    return (nitems / (nitems-1)) * (1 - (itemvars.sum() / tscores.var(ddof=1)))



dummy_fields = ['WEEKDAY','HOUR','level1_pnu','level2_pnu']

for dummy in dummy_fields:
    dummies = pd.get_dummies(df_total[dummy], prefix=dummy, drop_first=False)
    df_total = pd.concat([df_total, dummies], axis=1)

df_total = df_total.drop(dummy_fields,axis=1)



# 통계기법에서 LinearRegression
import statsmodels.api as sm

results = sm.OLS(train_y, train_x).fit()

results.summary()

# *** p<0.001, ** p<0.01, * p<0.05
# https://stats.stackovernet.xyz/ko/q/37406

# 기계학습에서 LinearRegression
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=lr()
model.fit(train_x, train_y)

print("모델의 회귀계수는 : ", model.coef_, "이고 모델의 절편은 : ",model.intercept_)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))



from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import GradientBoostingRegressor as grb
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

import pickle
import joblib
import time

model_list=[
            lr(),
            rfr(),
            grb(),
            xgb()
            ]

# 다차원 배열을 1차원으로 평평하게 만들어주기!
train_y = np.ravel(train_y, order='C')

model_rslt = []
for i in range(len(model_list)):
    start_time = time.process_time()
    model = model_list[i]
    model.fit(train_x, train_y)
    end_time = time.process_time()
    joblib.dump(model, '{}_model.pkl'.format(i)) # 모델 저장, sklearn을 통해서 만들어진 모델은 pkl 파일로 저장
    print(f"* {model} 결과 시작")
    print('----  {0:.5f}sec, training complete  ----'.format(end_time-start_time))
    pred_y = model.predict(test_x)
    model_rslt.append(model)
    print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
    print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))
    print("---------------------------------------------------------------------------")



# 모델 만들기 : 아주 간단한 모델

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_x.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])
  return model

# Case2 : 원핫인코딩 적용
print(tf.one_hot(temp_y[0], 10))
print(tf.one_hot(temp_y, 10))

# 모델 학습
early_stopping = EarlyStopping(monitor='val_loss', patience=10) # 조기종료 콜백함수 정의

checkpoint_path = 'tmp_checkpoint.ckpt'
cb_checkpoint = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor='val_loss',
                               verbose=1, save_best_only=True) # 체크포인트 저장

history = model.fit(train_x, train_y, epochs=30,
                   validation_data = (test_x,test_y),
                    callbacks=[cb_checkpoint, early_stopping]
                    )

# 최적 모델 불러오기 및 저장
model.load_weights(checkpoint_path)
model.save("DeeplearningModel.h5")

# Input_layer 가중치
model.layers[0].get_weights()[0]

# Input_layer 편향
model.layers[0].get_weights()[1]

# Dense 가중치
model.layers[1].get_weights()[0]

# Dense 편향
model.layers[1].get_weights()[1]

# output 가중치
model.layers[2].get_weights()[0]

# output 편향
model.layers[2].get_weights()[1]



# 모델을 담을 빈 리스트 생성
model_rslt = []

# 앞서 저장한 머신러닝 모델 불러오기 및 저장
for i in range(4):
    model_rslt.append(joblib.load("{}_model.pkl".format(i)))
# 앞서 저장한 딥러닝 모델 불러오기 및 저장
model_rslt.append(keras.models.load_model("DeeplearningModel.h5"))

e1_list = ['ETA1', 'ETA2', 'ETA3', 'ETA4', 'ETA5']
e2_list = ['ETAA1', 'ETAA2', 'ETAA3', 'ETAA4', 'ETAA5']

for e1, e2, model in zip(e1_list, e2_list, model_rslt):
    df_evaluation[e1] = model.predict(df_evaluation_feature)
    df_evaluation.loc[(df_evaluation[e1] < 0), e1] = 0
    etaa = (1-(abs(df_evaluation['ET']-df_evaluation[e1])/df_evaluation['ET']))*100.0
    df_evaluation[e2] = etaa
    df_evaluation.loc[(df_evaluation[e2] < 0), e2] = 0

# mean, min, max, std
etaa = ['ETAA', 'ETAA1', 'ETAA2', 'ETAA3', 'ETAA4', 'ETAA5']
alg = ['DATA', 'ML-LG', 'ML-RFR', 'ML-GBR', 'XBR', 'Deep']

print('+-------------------------------------------------------+')
print('|   ALG    | Mean(%) |    STD    |  MIN(%)  |  MAX(%)   |')
print('+----------+---------+-----------+----------+-----------+')
for i, e in zip(range(len(alg)), etaa):
    eMean = df_evaluation[e].mean()
    eStd = df_evaluation[e].std()
    eMin = df_evaluation[e].min()
    eMax = df_evaluation[e].max()
    print('|  {:6s}  |   {:3.1f}  |   {:05.1f}   |   {:4.1f}   |  {:7.1f}  | '.format(alg[i], eMean, eStd, eMin, eMax))
print('+----------+---------+-----------+----------+-----------+\n\n')



# 1단계 n-estimators와 learning_rate(eta)를 먼저 지정 : eta 0.1 => R-squared Score on Test set : 0.71631
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=xgb(n_estimators=100, eta=0.1)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))


# 1단계 n-estimators와 learning_rate(eta)를 먼저 지정 : eta 0.2 => R-squared Score on Test set : 0.71680
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=xgb(n_estimators=100, eta=0.2)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))


# 1단계 n-estimators와 learning_rate(eta)를 먼저 지정 : eta 0.3 => R-squared Score on Test set : 0.70988
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=xgb(n_estimators=100, eta=0.3)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))

# 2단계 나머지 hyperparameter 튜닝 : 0.71724
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=xgb(n_estimators=100, eta=0.2,
          max_depth=5, subsample= 0.8, colsample_bytree=0.5, reg_alpha=3, gamma=5)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))

# 3단계 n-estimators와 learning_rate 조정 : 0.71890
from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score

model=xgb(n_estimators=200, eta=0.1,
          max_depth=5, subsample= 0.8, colsample_bytree=0.5, reg_alpha=3, gamma=5)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)
print("RMSE on Test set : {0:.5f}".format(mean_squared_error(test_y,pred_y)**0.5))
print("R-squared Score on Test set : {0:.5f}".format(r2_score(test_y,pred_y)))

# 모델을 저장합니다.
import pickle
import joblib


joblib.dump(model, '4_model.pkl')



from xgboost import XGBRegressor as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import time

params = { 'n_estimators' : [50, 100, 200],
           'learning_rate' : [0, 0.01],
           'max_depth' : [0, 3],
            }

xgb_model = xgb(random_state = 0, n_jobs = 1)
grid_cv = GridSearchCV(xgb_model, param_grid = params, cv = 3, n_jobs = 1)
start_time = time.process_time()
grid_cv.fit(train_x, train_y)
end_time = time.process_time()

print('----  {0:.5f}sec, training complete  ----'.format(end_time-start_time))
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))



model_rslt = []
for i in range(5):
    model_rslt.append(joblib.load("{}_model.pkl".format(i)))
model_rslt.append(keras.models.load_model("DeeplearningModel.h5"))
model_rslt

etaa = ['ETAA', 'ETAA1', 'ETAA2', 'ETAA3', 'ETAA4', 'ETAA5', 'ETAA6']
alg = ['DATA', 'ML-LG', 'ML-RFR', 'ML-GBR', 'XBR', 'XBR2', 'Deep']

for length in range(4):
    if length == 0:
        print(' 1,000 <= A_DISTANCE < 5,000m')
    elif length == 1:
        print(' 5,000 <= A_DISTANCE < 10,000m')
    elif length == 2:
        print(' 10,000 <= A_DISTANCE < 15,000m')
    else:
         print('All A_DISTANCE')
    print('+-------------------------------------------------------+')
    print('|   ALG    | Mean(%) |    STD    |  MIN(%)  |  MAX(%)   |')
    print('+----------+---------+-----------+----------+-----------+')
    for i, e in zip(range(len(alg)), etaa):
        if length == 0:
            eMean = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=1000) & (df_evaluation['A_DISTANCE']<5000), e].mean()
            eStd = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=1000) & (df_evaluation['A_DISTANCE']<5000), e].std()
            eMin = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=1000) & (df_evaluation['A_DISTANCE']<5000), e].min()
            eMax = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=1000) & (df_evaluation['A_DISTANCE']<5000), e].max()
        elif length == 1:
            eMean = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=5000) & (df_evaluation['A_DISTANCE']<10000), e].mean()
            eStd = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=5000) & (df_evaluation['A_DISTANCE']<10000), e].std()
            eMin = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=5000) & (df_evaluation['A_DISTANCE']<10000), e].min()
            eMax = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=5000) & (df_evaluation['A_DISTANCE']<10000), e].max()
        elif length == 2:
            eMean = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=10000) & (df_evaluation['A_DISTANCE']<15000), e].mean()
            eStd = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=10000) & (df_evaluation['A_DISTANCE']<15000), e].std()
            eMin = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=10000) & (df_evaluation['A_DISTANCE']<15000), e].min()
            eMax = df_evaluation.loc[(df_evaluation['A_DISTANCE']>=10000) & (df_evaluation['A_DISTANCE']<15000), e].max()
        else:
            eMean = df_evaluation[e].mean()
            eStd = df_evaluation[e].std()
            eMin = df_evaluation[e].min()
            eMax = df_evaluation[e].max()
        print('|  {:6s}  |   {:3.1f}  |   {:05.1f}   |   {:4.1f}   |  {:7.1f}  |'.format(alg[i], eMean, eStd, eMin, eMax))
    print('+----------+---------+-----------+----------+-----------+\n\n')
