# MindScope
Stress Prediction Model (Server part)

<img width="682" alt="그림1" src="https://user-images.githubusercontent.com/25919167/97527692-47f44100-19ef-11eb-81be-3b6e5be516d6.png">
**Overall Flow of Project**

1) 데이터 수집 기간 : 약 1~2주간 사용자의 센서 데이터 & 스트레스 상태 수집(PSS 척도)
2) 스트레스 예측 모델 Initializing : 데이터 수집 종료 이후, 수집된 데이터 기반으로 각 사용자별 스트레스 예측 모델 초기화
3) 스트레스 에측 및 평가
  - 데이터 수집 종료 이후, 초기화된 스트레스 예측 모델을 기반으로 하루 4번 스트레스 예측
  - 스트레스 예측 결과 및 스트레스 예측에 영향을 미친 Feature 정보 제공
  - 스트레스 예측 결과에 대한 사용자의 피드백 수집

**Role of Stress Prediction**

- Model : 사용자의 스마트폰 센서 데이터 기반으로 4시간마다 사용자의 스트레스 예측
- XAI (SHAP) : 스트레스 예측에 영향을 미친 feature 값 정보를 보여줌

**Main Flow of Stress Prediction Model** [Code Link](https://github.com/sy2399/MindScope/blob/master/main_service/stress_prediction_service.py)

- main_service > stress_prediction_service.py
  1) service_routine() : 4시간 마다 gRPC 스트레스 예측 모델 프로세스 실행 

**Main Module of Stress Prediction Model** [Code Link](https://github.com/sy2399/MindScope/blob/master/main_service/stress_model.py)

- main_service > stress_model.py
