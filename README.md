# OSSP

Heart Attack Analysis & Prediction(심장마비 분석, 예측)

데이터 : https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

참고한 오픈소스 : https://www.kaggle.com/code/advikmaniar/heart-attack-eda-prediction-with-9-model-95/notebook

(License : This Notebook has been released under the Apache 2.0 open source license.)




### 1. 데이터 설명
---
입력 데이터에 대한 설명입니다.
- Age : 환자의 나이
- Sex : 환자의 성별 (1,0)
  - 데이터셋에서 성별을 1과 0으로 표현하였는데 어떤 것이 여자인지 남자인지에 대한 것은 설명하지 않았다.
- exng : 운동 유발 협심증 (1 = 예, 0 = 아니오)
  - 협심증 : 심장에 혈액을 공급하는 혈관인 관상 동맥이 동맥 경화증으로 좁아져서 생기는 질환, 운동 등의 신체활동을 할 때 나타남
- ca : 혈관의 수(0 ~ 3)
- cp : 가슴 통증의 타입
  - 0 : 전형적인 협심증
  - 1 : 비정형 협심증
  - 2 : 협심증이 아닌 통증
  -  3 : 증상 X
- trtbps :  안정된 상태에서의 혈압
- chol : 혈중 콜레스테롤(mg/dl)
- fbs : 공복 혈당(혈당 > 120mg/dl == true : 1, false : 0)
- rest_ecg : 심전도
  - 0 : 정상 
  - 1 : ST-T파 이상(T파 역전 및/또는 > 0.05mV의 ST 상승 또는 하강)
  - 2 : Estes 기준에 따라 가능성이 있거나 확실한 좌심실 비대
- thalachh : 최대 심박수
- slp : 비교적 안정되기까지 운동으로 유발되는 ST 우울증
  - ST depression : 심전도 결과에서 ST 세그먼트를 나타내는 용어
- oldpeak : 최대 운동 St segment의 기울기
- thall : 지중해성 빈혈
- output
  - 1 : 심장마비를 경험할 확률이 높은 사람
  - 0 : 심장마비를 경험할 확률이 낮은 사람


- 연속형 데이터(Continuous data) : age, trtbps, chol, thalach, oldpeak
- 분류형 데이터(Classification data) : sex, output, cp, fbs, exng, restecg, thall, caa, slp




### 2. 라이브러리 추가
``` python
import numpy as np # 행렬, 다차원 배열을 다룰 때 사용
import pandas as pd # 데이터를 다룰 때 사용
import seaborn as sns #matlab을 기반으로 한 시각화
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')3

from google.colab import drive
drive.mount('/content/Kaggle_Heart_Attack_data')
```




### 3. 데이터 불러오기
``` python
# data download : https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download

data = pd.read_csv("/content/Kaggle_Heart_Attack_data/MyDrive/Kaggle_Heart_Attack_data/heart.csv")

data.head()
```

|   |age|sex|cp|trtbps|chol|fbs|restecg|thalachh|exng|oldpeak|slp|caa|thall|output|
|-----|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|63|1|3|145|233|1|0|150|0|2.3|0|0|1|1|
|1|37|1|2|130|250|0|1|187|0|3.5|0|0|2|1|
|2|41|0|1|130|204|0|0|172|0|1.4|2|0|2|1|
|3|56|1|1|120|236|0|1|178|0|0.8|2|0|2|1|
|4|57|0|0|120|354|0|1|163|1|0.6|2|0|2|1|

``` python
display(data.describe())
display(data.info())
```
![image](https://user-images.githubusercontent.com/121947465/211258105-29443719-9c7d-4575-831d-8a498a8b6c7c.png)

![image](https://user-images.githubusercontent.com/121947465/211258133-92a8936d-66c1-43b3-837e-c9e4b1c2ebf0.png)




### 4. 데이터 분석

- 성별 
성별 데이터 '0', '1'이 각각 여성인지 남성인지 분별하기 위해 성별에 대한 분석을 우선 수행합니다.

심장질환은 여성보다 남성에게 일어날 확률이 높다는 것을 여러 학술자료를 통해 알 수 있으므로 이것을 기반으로 분석합니다.

``` python
output_graph1=px.pie(data, names= "sex",title="sex")
output_graph1.show()
output_graph2=px.pie(data, names= "output",title="Output")
output_graph2.show()

print("Female Value Counts: \n{}".format((data[data ["sex"] == 1].reset_index())['output'].value_counts()))
print("Male Value Counts: \n{}".format((data[data ["sex"] == 0].reset_index())['output'].value_counts()))
```
![image](https://user-images.githubusercontent.com/121947465/211483715-cc422d5e-0ce5-4a44-b978-4db26065c62c.png)
![image](https://user-images.githubusercontent.com/121947465/211483886-83c72a42-779a-40cc-a216-d8b5975a8b4b.png)
![image](https://user-images.githubusercontent.com/121947465/211483995-fa740f72-7931-4d61-9c60-e447b7eadbe9.png)

총 303개의 데이터 중 207명(68.3%)은 1, 96명(31.7%)은 0입니다.
``` python
fig, ax1 = plt.subplots(1,2, figsize=(20,6))
plt.suptitle("Female(1)                                                                                                                     Male(0)")
sns.countplot("output", data=X, palette='gist_heat',ax=ax1[0])
sns.countplot("output", data=Y, palette='gist_heat',ax=ax1[1])
fig.show()
```
![image](https://user-images.githubusercontent.com/121947465/213958347-402ded4b-6f35-4b4a-b7ec-084a80221f74.png)

데이터가 '0'인 성별은 전체 데이터 비율이 데이터가 '1'인 성별보다 낮고, output의 비율은 1(심장질환을 경험할 확률이 높은 사람)이 더 높게 나타납니다.

데이터가 '1'인 성별은 전체 데이터 비율이 데이터가 '0'인 성별보다 높고, ouput의 비율은 0(심장질환을 경험할 확률이 낮은 사람)이 더 높게 나타납니다.

-> 성별 데이터 '0'이 남성, '1'이 여성인 것을 유추할 수 있습니다.



- 나이 분석
``` python
plt.figure(figsize=(18,6))
sns.distplot(data["age"], color="magenta")
plt.title("Total Age distribution")
plt.show()
```
![image](https://user-images.githubusercontent.com/121947465/213958429-cb9f063f-0cea-4a0e-986d-ded70bb9f18a.png)

나이의 분포 그래프입니다. 50 ~ 60대가 대부분입니다.


``` python
ax = px.histogram(data, x="age", color="output", title="Distribution Age and Output")
ax.show()
```
![image](https://user-images.githubusercontent.com/121947465/212066767-f84b5ba8-c3b2-4ba7-8a15-b83ac097e5f4.png)
나이와 Output의 관계입니다.

``` python
X=data[data["sex"]==1].reset_index()  # 여성
Y=data[data["sex"]==0].reset_index()   # 남성

HR=data[data["output"]==1].reset_index()  # Output=1(심장질병에 걸릴 확률이 높은 데이터)
LR=data[data["output"]==0].reset_index()  # Output=0(심장질병에 걸릴 확률이 낮은 데이터)

print("----------------나이 분포------------------")
print("평균: {}".format(round(data["age"].mean())))
print("중앙값: {}".format(round(data["age"].median())))
print("분산: {}".format(round(data["age"].var())))  
print("표준편차: {}\n".format(round((data["age"].std()),3)))


print("--------------성별에 따른 나이 분포-------------------")
print("-남성-")
print("평균: {}".format(round(Y["age"].mean())))
print("중앙값: {}".format(round(Y["age"].median())))
print("분산: {}".format(round(Y["age"].var())))
print("표준편차: {}\n\n".format(round((Y["age"].std()),3)))

print("-여성-")
print("평균: {}".format(round(X["age"].mean())))
print("중앙값: {}".format(round(X["age"].median())))
print("분산: {}".format(round(X["age"].var())))
print("표준편차: {}\n".format(round((X["age"].std()),3)))


print("------------------Output에 따른 나이 분포-----------------------")
print("-심장질병에 걸릴 확률이 높은 나이-")
print("평균: {}".format(round(HR["age"].mean())))
print("중앙: {}".format(round(HR["age"].median())))
print("분산: {}".format(round(HR["age"].var())))
print("표준편차: {}\n\n".format(round((HR["age"].std()),3)))

print("-심장질병에 걸릴 확률이 낮은 나이-")
print("평균: {}".format(round(LR["age"].mean())))
print("중앙값: {}".format(round(LR["age"].median())))
print("분산: {}".format(round(LR["age"].var())))
print("표준편차: {}".format(round((LR["age"].std()),3)))
```
![image](https://user-images.githubusercontent.com/121947465/212068582-c878809d-a060-4846-8600-1ea5318072d4.png)

- 나이와 혈압 점 그래프
X축 : 나이, Y축 : 혈압

![image](https://user-images.githubusercontent.com/121947465/212070351-831f202a-bf09-4b77-86ce-7e895953d2df.png)

``` python
# 분류형, 연속형 데이터 나누기
class_cols=["sex","output",'cp',"fbs","exng","restecg","thall","caa","slp"]
class_data=data[class_cols]

continuous_cols=["age","trtbps","chol","thalachh","oldpeak"]
continuous_data=data[continuous_cols]


# 분류형 데이터 그래프
for col in class_cols[2:]:
    ax=px.pie(data, names= col, title=col)
    ax.show()


# 연속형 데이터 그래프
continuous = ["age","trtbps","chol","thalachh","oldpeak", "output"]
fig, ax1 = plt.subplots(2,3, figsize=(20,20))
k = 0
for i in range(2):
  for j in range(3):
    sns.distplot(data[continuous[k]], ax = ax1[i][j], color = 'red')
    k +=1

plt.show()
```
![image](https://user-images.githubusercontent.com/121947465/212470131-70a96ab7-fbd8-43d8-a191-0a4a5f3fe90a.png)
![image](https://user-images.githubusercontent.com/121947465/212470135-f3c2fe41-50fc-46bb-b691-f7287189e202.png)
![image](https://user-images.githubusercontent.com/121947465/212470137-5cc4ef40-f6b0-45da-8703-81d355d8ac7f.png)
![image](https://user-images.githubusercontent.com/121947465/212470140-788fcf97-2c51-4f9b-ac5e-b155454911c1.png)
![image](https://user-images.githubusercontent.com/121947465/212470143-4119ff78-1350-4514-bc3a-bc427b5c7862.png)
![image](https://user-images.githubusercontent.com/121947465/212470145-82e9eb1d-a9e1-4e8e-922d-38ea58ff976c.png)
![image](https://user-images.githubusercontent.com/121947465/212470150-b3ac9443-7dbb-4bfb-9a84-5cfd5944cbf7.png)


![image](https://user-images.githubusercontent.com/121947465/212469953-290728cc-287a-4c05-867e-bf2f43dec3fb.png)
![image](https://user-images.githubusercontent.com/121947465/212469963-72154d23-08df-4a50-a907-1051ef041c60.png)




### 5. 데이터 전처리
###### - 결측치는 데이터셋에 존재하지 않으므로 고려하지 않습니다.

- 이상치 탐색

``` python
fig, ax1 = plt.subplots(2,2, figsize=(20,12))
k = 0
for i in range(2):
    for j in range(2):
        sns.boxplot(data=data,x=data[continuous_cols[1:][k]],saturation=1,ax=ax1[i][j],color="white")
        k+=1
plt.tight_layout()
plt.show()

# 이상치 탐색
'''
# 이상치 판별, 시각화

Q3 = data["trtbps"].quantile(q=0.75)
Q1 = data["trtbps"].quantile(q=0.25)
IQR = Q3 - Q1

print("이상치(최댓값 초과) : ",Q3 + IQR*1.5)
print("이상치(최솟값 미만) : ", Q1 - IQR*1.5)
'''

# Display the position of outliners.
print("Outliners Present at position: \n")
print("trtbps: {}".format(np.where(data['trtbps']>170)))
print("chol: {}".format(np.where(data['chol']>369.75)))
print("thalachh: {}".format(np.where(data['thalachh']<84.75)))
print("oldpeak: {}".format(np.where(data['oldpeak']>4)))
```

![image](https://user-images.githubusercontent.com/121947465/213958556-34371312-6175-4c4b-9f2c-dd0a19c944e7.png)

![image](https://user-images.githubusercontent.com/121947465/213958604-98b23e36-9aa8-4f04-857e-5f8e628de7d6.png)

박스 도표를 통해 데이터의 전반적인 구성을 보고 이상치를 판별합니다.
총 이상치는 24개입니다. (중복되는 101, 220, 223은 한번만 포함시켰.)

- 이상치 제거

참고 사이트는 이상치 제거를 '로그 변환'을 통해 수행했습니다. 로그 변환으로 이상치를 제거할 경우 이상치가 완전히 제거되진 않습니다.

저는 IQR 방법을 사용하여 이상치를 완전히 제거하여 두 방법을 비교해보도록 하겠습니다.

``` python
# 이상치 제거(IQR)

import copy

continuous_cols=["age","trtbps","chol","thalachh","oldpeak"]
continuous_data=data[continuous_cols]
Outliner_delete_data = copy.deepcopy(data)

a=1
for a in range(1, 5):
  Q3 = data[continuous_cols[a]].quantile(q=0.75)
  Q1 = data[continuous_cols[a]].quantile(q=0.25)
  IQR = Q3 - Q1
  outliner_max = data[continuous_cols[a]] > Q3 + IQR*1.5  # 이상치(최댓값 초과)
  outliner_min = data[continuous_cols[a]] < Q1 - IQR*1.5  # 이상치(최솟값 미만)
  outliner_max_index = data[outliner_max].index  # 인덱스
  outliner_min_index = data[outliner_min].index  # 인덱스
  
  print(outliner_max_index)
  print(outliner_min_index)
  #print("----------------------------------------------")
  

  Outliner_delete_data.drop(outliner_max_index, inplace=True, errors='ignore')
  Outliner_delete_data.drop(outliner_min_index, inplace=True, errors='ignore')


fig, ax1 = plt.subplots(2,2, figsize=(20,12))
k = 0
for i in range(2):
    for j in range(2):
        sns.boxplot(data=Outliner_delete_data,x=Outliner_delete_data[continuous_cols[1:][k]],saturation=1,ax=ax1[i][j],color="white")
        k+=1
plt.tight_layout()
plt.show()

display(Outliner_delete_data.info())
```

결측치를 제거한 데이터셋의 박스도표
    
    

![image](https://user-images.githubusercontent.com/121947465/212476631-96ca56b4-4215-455b-8028-2b3a4cbf5548.png)
![image](https://user-images.githubusercontent.com/121947465/212474582-de320939-7c33-438c-b084-dd237f17e744.png)
![image](https://user-images.githubusercontent.com/121947465/212476635-446869da-2395-4087-a192-efa4234fdb8f.png)

303개의 데이터 중 이상치 19개가 제거되었습니다.

``` python
# 이상치 제거(로그 변환)
data["age"]= np.log(data.age)
data["trtbps"]= np.log(data.trtbps)
data["chol"]= np.log(data.chol)
data["thalachh"]= np.log(data.thalachh)
print("---Log Transform performed---")

continuous_cols=["age","trtbps","chol","thalachh","oldpeak"]
continuous_data=data[continuous_cols]

for k, v in continuous_data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column {} outliers = {} => {}%".format(k,len(v_col),round((perc),3)))
```
![image](https://user-images.githubusercontent.com/121947465/212544139-af4423ec-c166-42d7-96c1-fea89247e13e.png)

전체 이상치가 로그변환을 통해 값들이 작아지면서 줄어들었습니다.

###### - 이상치를 탐색하고 수정하는 것은 연속형 데이터에만 해당합니다. 분류형 데이터는 one-hot 인코딩 방식을 사용하여 Accuracy를 비교해 보겠습니다.
``` python
# one-hot encoding
cat_col = ['sex','cp','fbs','restecg', 'exng', 'slp', 'caa', 'thall']
data[cat_col] = data[cat_col].astype('category')
data = pd.get_dummies(data, columns=cat_col, drop_first=True)
display(data)
```
![image](https://user-images.githubusercontent.com/121947465/212817409-17220deb-b29f-4fb2-a5aa-4a8739dbc06f.png)


##### - 학습, 평가용 데이터 분류 / 정규화
  

``` python
# 데이터 분할(IQR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = Outliner_delete_data.iloc[:,:13]
Y = Outliner_delete_data["output"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 1)

MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler.fit_transform(X_test)
```

``` python
# 데이터 분할(로그 변환 데이터)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X=data.iloc[:,:13]
Y=data["output"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65) 

#MinMax Scaling / Normalization of data
MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler.fit_transform(X_test)

```
``` python
# 데이터 분할(one-hot encoding 데이터)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
Y = data["output"]
X = data.drop("output", axis=1)
display(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65) 

#MinMax Scaling / Normalization of data
MM_scaler = MinMaxScaler()
X_train = MM_scaler.fit_transform(X_train)
X_test = MM_scaler.fit_transform(X_test)
```
![image](https://user-images.githubusercontent.com/121947465/212817681-9b8e4145-0f4e-40fc-ada2-7023db0f280a.png)
  
  
  
   
   
##### - 예측 결과와 정확도 시각화 함수
  
``` python
def compute(Y_pred,Y_test):
    # 각각의 output을 잘 예측했는지 시각화
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(Y_pred)),Y_pred,color="yellow",lw=5,label="Predictions")
    plt.scatter(range(len(Y_test)),Y_test,color="red",label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()
    plt.show()

    # 혼동 행렬을 통한 예측 정확도 시각화
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # 정확도 계산
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
    print('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Square Error: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round((acc*100),3), round((mse),3)))
 ```


### 6. 예측
##### 6-1. 로지스틱 회귀
``` python
# 1. Build Model(Logistic Regression)
start = time.time()

model_Log= LogisticRegression(random_state=10)
model_Log.fit(X_train,Y_train)
Y_pred= model_Log.predict(X_test)

end=time.time()

model_Log_time=end-start
model_Log_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_Log_time),5)} seconds\n")
#Plot and compute metrics
compute(Y_pred,Y_test)
```
- 예측 정확도 : 82.456%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212822939-0d645c73-0499-4ab0-8f50-c99a01d8ab83.png)

- 예측 정확도 : 90.164%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212544973-b6ce8c4f-021c-4f0a-94c2-289bc0537788.png)


##### 6-2. K-최근접 이웃
``` python
# 2. Build Model(K-Nearest Neighbours)
from sklearn.neighbors import KNeighborsClassifier
start=time.time()

model_KNN = KNeighborsClassifier(n_neighbors=15)
model_KNN.fit(X_train,Y_train)
Y_pred = model_KNN.predict(X_test)

end=time.time()

model_KNN_time = end-start
model_KNN_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_KNN_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 78.947%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821327-6c484c3a-815b-4f33-8e82-e7ed493fcc73.png)

- 예측 정확도 : 88.525(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212681111-59a0ad6a-eb97-42fa-8f11-ec736a1bc5e0.png)

##### 6-3. 서포트 벡터 머신
``` python
# 3. Build Model(Support Vector Machine)
from sklearn.svm import SVC


start=time.time()

model_svm=SVC(kernel="rbf")
model_svm.fit(X_train,Y_train)
Y_pred=model_svm.predict(X_test)

end=time.time()

model_svm_time=end-start
model_svm_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_svm_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 78.947%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821483-159ec3ef-a323-4e08-8791-639f93c1ea72.png)

- 예측 정확도 : 90.164%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212681740-8ecfcb6f-2968-49ae-b51f-c92ff21cbf2e.png)

##### 6-4. 의사결정트리
``` python
from sklearn.tree import DecisionTreeClassifier
# 4. Build Model(Decision Tree)
start=time.time()

model_tree=DecisionTreeClassifier(random_state=10,criterion="gini",max_depth=100)
model_tree.fit(X_train,Y_train)
Y_pred=model_tree.predict(X_test)

end=time.time()

model_tree_time=end-start
model_tree_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_tree_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 73.684%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821578-92cbdbe2-517c-435d-b0e7-96de9d580276.png)

- 예측 정확도 : 81.967%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212681991-ffadc878-43f2-461e-8595-e832ef03c0d3.png)

##### 6-5. 랜덤 포레스트 분류
``` python
from sklearn.ensemble import RandomForestClassifier
# 5. Build Model(RandomForest)
start=time.time()

model_RF = RandomForestClassifier(n_estimators=300,criterion="gini",random_state=5,max_depth=100)
model_RF.fit(X_train,Y_train)
Y_pred=model_RF.predict(X_test)

end=time.time()

model_RF_time=end-start
model_RF_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_RF_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 77.193%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821678-2c221b7d-c359-44af-8f0e-e0fec7f0a88b.png)

- 예측 정확도 : 91.803%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212682468-52595c24-1927-4e5c-bc02-ed6a3347bb8e.png)

##### 6-6. AdaBoost
``` python
from sklearn.ensemble import AdaBoostClassifier
# 6. Build Model(AdaBoost)
start=time.time()

model_ADA=AdaBoostClassifier(learning_rate= 0.15,n_estimators= 25)
model_ADA.fit(X_train,Y_train)
Y_pred= model_ADA.predict(X_test)

end=time.time()

model_ADA_time=end-start
model_ADA_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_ADA_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 80.702%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821753-aed9f735-02e3-4c56-8e57-087e9756bf39.png)

- 예측 정확도 : 93.443%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212682977-181567df-1403-461e-a7e8-dc2c1f866ec0.png)

##### 6-7. Gradient Boosting
``` python
from sklearn.ensemble import GradientBoostingClassifier
# 7. Build Model(Gradient Boosting)
start=time.time()

model_GB= GradientBoostingClassifier(random_state=10,n_estimators=20,learning_rate=0.29,loss="deviance")
model_GB.fit(X_train,Y_train)
Y_pred= model_GB.predict(X_test)

end=time.time()

model_GB_time=end-start
model_GB_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_GB_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 77.193%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821827-72b3c9f9-51e6-4895-ae4c-41dd8e53ef10.png)

- 예측 정확도 : 91.803%()

![image](https://user-images.githubusercontent.com/121947465/212683418-72556ff0-423d-42f9-a1d2-dee490b34b11.png)


##### 6-8. XGBoost
``` python
from xgboost import XGBClassifier
# 8. Build Model(XG Boost)
start=time.time()

model_xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.1,
                          max_depth=1,
                          n_estimators = 50,
                          colsample_bytree = 0.5)
model_xgb.fit(X_train,Y_train)
Y_pred = model_xgb.predict(X_test)

end=time.time()

model_xgb_time=end-start
model_xgb_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_xgb_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 84.211%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212821922-5f557303-f52c-416f-a9b5-e3ae5ab03cc8.png)

- 예측 정확도 : 95.082%(로그변환)

![image](https://user-images.githubusercontent.com/121947465/212683649-3c929def-3def-43aa-a0b5-34545ce852a2.png)


##### 6-9. MLP
``` python
# 9. Build Model(MLP)
from sklearn.neural_network import MLPClassifier

start=time.time()

model_MLP = MLPClassifier(random_state=48,hidden_layer_sizes=(150,100,50), max_iter=150,activation = 'relu',solver='adam')
model_MLP.fit(X_train, Y_train)
Y_pred=model_MLP.predict(X_test)

end=time.time()

model_MLP_time=end-start
model_MLP_accuracy=round(accuracy_score(Y_test,Y_pred), 4)*100 # Accuracy

print(f"Execution time of model: {round((model_MLP_time),5)} seconds")
#Plot and compute metric
compute(Y_pred,Y_test)
```
- 예측 정확도 : 78.947%(IQR)

![image](https://user-images.githubusercontent.com/121947465/212822496-18a9de9d-d7b1-4fc5-8efb-1ff204ade7f0.png)


###### - 모든 모델에서 로그변환을 사용한 모델이 IQR을 사용한 모델보다 정확도가 높은 것을 볼 수 있습니다.
###### - 정확도를 조금 더 높이기 위해 One-hot encoding과 GridSearch를 이용한 하이퍼파라미터의 조정을 통해 원래 모델과 비교해보겠습니다.


- 기존 모델 정확도 비교
![image](https://user-images.githubusercontent.com/121947465/213193674-54af0037-4808-430a-9518-68ca1d748898.png)

- 로그변환 + 하이퍼파라미터를 조정한 모델들의 정확도
-> Logistic regression, KNN, Decision Tree, Random Forest, Gradient Boosting 모델의 정확도가 향상되었습니다.
![image](https://user-images.githubusercontent.com/121947465/213193718-d0813520-9e41-41fe-bc7b-385e96a6c943.png)

- 로그변환 + One-hot encoding + 하이퍼파라미터를 조정한 모델들의 정확도
-> Logistic regression, KNN, SVM의 정확도가 올랐고, MLP Classifier 모델의 정확도가 기존 소스의 최고 정확도인 95.082%를 넘어 96.72%로 향상되었습니다.
![(정확도높음)로그변환+One_hot+하이퍼파라미터](https://user-images.githubusercontent.com/121947465/213193897-f2ec086a-b40a-46ae-817d-b0defdec68f3.png)
