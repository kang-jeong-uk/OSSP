# OSSP

Heart Attack Analysis & Prediction(심장마비 분석, 예측)

데이터 : https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

### 1. Data Description
---

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
  
  
- slp : 비교적 안정되기까지 운동으로 유발되는 ST depression
- oldpeak : 최대 운동 St segment의 기울기

### 2. Library import
``` python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as msno
import plotly.graph_objs as go
import plotly.express as px
plt.style.use('seaborn-dark')
plt.style.context('grayscale')
```
    
