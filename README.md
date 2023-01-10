# OSSP

Heart Attack Analysis & Prediction(심장마비 분석, 예측)

데이터 : https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

### 1. 데이터 설명
---
입력 데이터에 대한 설명입니다.


- Age : 환자의 나이
- Sex : 환자의 성별 (1,0)
  - 데이터셋은 성별을 1과 0으로 표현하였는데 어떤 것이 여자인지 남자인지에 대한 것은 설명하지 않았다.
- exang : 운동 유발 협심증 (1 = 예, 0 = 아니오)
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
  - 2 : Estes의 기준에 따라 가능성이 있거나 확실한 좌심실 비대
- thalach : 최대 심박수
- target : 
  - 0 : 심장마비 가능성 낮음 
  - 1 : 심장마비 가능성 높음
  
  
- slp : 비교적 안정되기까지 운동으로 유발되는 ST 우울증
  - ST depression : 심전도 결과에서 ST 세그먼트를 나타내는 용어
- oldpeak : 최대 운동 St segment의 기울기
- output
  - 1 : 심장마비를 경험할 확률이 높은 사람
  - 0 : 심장마비를 경험할 확률이 낮은 사람
### 2. 라이브러리 추가
``` python
import numpy as np # 행렬, 다차원 배열을 다룰 때 사용
import pandas as pd # 데이터를 다룰 때 사용
import seaborn as sns #matlab을 기반으로 한 시각화
import matplotlib.pyplot as plt
import missingno as msno # 결측값을 확인할 때 사용
import plotly.graph_objs as go
import plotly.express as px
plt.style.use('fivethirtyeight')
plt.style.use('seaborn-dark')

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
``` python
plt.figure(figsize=(20,10))
sns.pairplot(data, hue="output", corner = True)
plt.legend("output")
plt.tight_layout()
plt.plot()
```
![image](https://user-images.githubusercontent.com/121947465/211485170-ac80423a-7ed7-4d04-999a-935627099a9d.png)

성별 데이터 '0', '1'이 각각 여자인지 남자인지 분별하기 위해 성별에 대한 분석을 우선 수행합니다.
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

![image](https://user-images.githubusercontent.com/121947465/211488720-fbe2c6e2-a2a4-44b0-9287-daad5c73cdc0.png)

데이터가 '0'인 성별은 전체 데이터 비율이 1보다 낮고, output의 비율은 1(심장질환을 경험할 확률이 높은 사람)이 더 높게 나타납니다.

데이터가 '1'인 성별은 전체 데이터 비율이 0보다 높고, ouput의 비율은 0(심장질환을 경험할 확률이 낮은 사람)이 더 높게 나타납니다.

-> 성별 데이터 '0'이 여자, '1'이 남자인 것을 유추할 수 있습니다.
