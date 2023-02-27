# DACON 기계 고장 진단 AI 경진대회 [(Link)](https://dacon.io/competitions/official/236036/overview/description)

---

## 대회 소개

### 목표

* 음향 데이터으로부터 비지도 학습을 통해 기계 고장을 진단하는 AI 알고리즘 개발

### 참여 기간

* 2022.12.05 ~ 2023.01.16

### 참여자

* 권남우

### 사용한 Tool
* Python
* Pandas / Scikit - Learn / time / datetime / librosa / matplotlib / seaborn
* Colab


### 결과

* PUBLIC SCORE : 0.98224 (4등/219팀)
* PRIVATE SCORE : 0.93957 (18등/219팀)
* EDA 좋아요 28개
* Feature Extraction in Audio (Sound) 좋아요 15개

---

## 데이터셋 소개

### Train Set
* 총 1279 개 팬(FAN) 소리 샘플
* FAN_TYPE : 팬(FAN)의 모델 종류 (0, 2 존재)
* LABEL (전부 0 : 정상 데이터만 존재)

### Test Set
* 총 1514 개 팬(FAN) 소리 샘플
* FAN_TYPE : 팬(FAN)의 모델 종류 (0, 2 존재)

---

## 프로세스

### Understading Domain (Audio)
* 효과적인 Feature Extraction을 수행하기 위해서는 음향 데이터에 대한 충분한 이해가 필요했다.
* 아날로그 신호, 진폭, 주파수, 위상, 대역폭, 스팩트럼 등과 같은 기본적인 개념과 wav 파일로 부터 추출하는 있는 Feature들의 개념을 공부하였다.

### EDA
* 모든 파일이 동일한 Sampling Rate를 가지는 것을 확인하였다.
* Fan Type 별로 3개의 샘플을 선택하여 다양한 Feature들을 시각화하였다.
* 특히 MFCC로 Feature Extraction를 수행한 결과를 히스토그램과 박스 플랏을 표현해보았을 때, Fan Type 별로 분포가 다르다는 것을 확인할 수 있었다.

### Preprocessing

#### Feature Extraction
* MFCC를 기반으로 전진선택법을 통해 다양한 Feature들을 추가하였다. (Zero Crossing Rate, RMS, Poly Feature, Spectral Flatness)
* 추출된 Feature들로 부터 좀 더 다양한 정보를 얻기 위해 Zero Crossing Rate과 RMS를 각각 차분한 값을 추가로 사용하였다.

#### Scaling
* Fan Type별로 데이터 분포가 크게 차이나였기에 Fan Type 구분하여 Scaling를 적용하였다. 
* 이상치가 존재함을 시각화를 통해 확인하였으며 Robust Scaler를 적용하였다.
* Robust Scaler의 하이퍼 파리미터 'quantile_range'를 (15.0, 85.0)로 설정하였을 때, Public Score가 상승하였다.

#### Dimension Reduction
* 차원 축소를 통해 고차원 데이터를 저차원 공간에 투영해 중복 정보를 제거하면서 가능한 핵심 정보 유지하고자 하였다.
* SparsePCA를 사용하여 희소성을 유지함으로써 PCA보다 더 좋은 성능을 얻을 수 있었다.

### Modeling

* Local Outlier Factor 모델을 사용하였다.
* 모델 특성상 Novelty Score가 정규화 되어있지 않기에 해당 모델을 다른 Data Set에 적용하는 것은 좋지 못할 수 있다.
* 그래서 그래서 FAN TYPE 별로 각기 다른 Data Set이라고 가정하여 AN TYPE 별로 각각 모델을 학습하였다.
* 
---

## 느낀 점
