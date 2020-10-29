# MindScope
MyPart : Stress Prediction Model (Server part)

## 프로젝트 소개 [Link](http://haesookim.info/MindScope/index.html)

<p align="center"><img width="676" alt="mindscope-intro" src="https://user-images.githubusercontent.com/25919167/97529377-21d0a000-19f3-11eb-9360-98c0d0579617.png"></p>
<p align="center"><img width="842" alt="experiment" src="https://user-images.githubusercontent.com/25919167/97529382-2301cd00-19f3-11eb-80dd-27aa3f9b3bd7.png"></p>

**Overall Flow of Project**

1) 데이터 수집 기간 : 약 1~2주간 사용자의 센서 데이터 & 스트레스 상태 수집(PSS 척도)
2) 스트레스 예측 모델 Initializing : 데이터 수집 종료 이후, 수집된 데이터 기반으로 각 사용자별 스트레스 예측 모델 초기화
3) 스트레스 에측 및 평가
  - 데이터 수집 종료 이후, 초기화된 스트레스 예측 모델을 기반으로 하루 4번 스트레스 예측
  - 스트레스 예측 결과 및 스트레스 예측에 영향을 미친 Feature 정보 제공
  - 스트레스 예측 결과에 대한 사용자의 피드백 수집
  <p align="center"><img width="397" alt="그림2" src="https://user-images.githubusercontent.com/25919167/97528300-b84f9200-19f0-11eb-9c73-b0da491c18a9.png"></p>

***

## My Part : Stress Prediction
<p align="center"><img width="682" alt="그림1" src="https://user-images.githubusercontent.com/25919167/97527692-47f44100-19ef-11eb-81be-3b6e5be516d6.png"></p>

**Role of Stress Prediction**

- Model : 사용자의 스마트폰 센서 데이터 기반으로 4시간마다 사용자의 스트레스 예측
- XAI (SHAP) : 스트레스 예측에 영향을 미친 feature 값 정보를 보여줌

**Main Flow of Stress Prediction Model** [Code Link](https://github.com/sy2399/MindScope/blob/master/main_service/stress_prediction_service.py)

- main_service > stress_prediction_service.py
  1) service_routine() : 4시간 마다 gRPC 스트레스 예측 모델 프로세스 실행  ==> prediction_task() 호출
  2) prediction_task()
    - 데이터 수집 기간 동안은 아무 기능을 하지 않음
    - 데이터 수집 종료 시점 : 모델 초기화 ==> initialModelTraining() 호출
    - 스트레스 예측 기간 
      - 4시간 마다, gRPC 로부터 사용자의 스마트폰 센서 데이터 불러와 테스트 케이스로 사용 ==> grpc_handler.grpc_load_user_data()
      - 모델의 예측값 및 모델에 영향을 미친 feature (SHAP) 값 저장 및 gRPC 서버로 전송 ==> initModel.predict() & StressModel.saveAndGetSHAP()
      - 모델의 예측값에 대한 사용자의 피드백을 받아 모델 재학습 ==> StressModel.update()

**Main Module of Stress Prediction Model** [Code Link](https://github.com/sy2399/MindScope/blob/master/main_service/stress_model.py)

- main_service > stress_model.py
  - class StressModel
    - mapLabel() : Data Labeling (STRESS LEVEL : 낮음, 보통, 매우 높음)
    - preprocessing() : 데이터 전처리
    - normalizing() : MinMax Normalizing 
    - initModel() : 모델 초기화 - RandomForest
    - saveAndGetSHAP() : 모델의 예측값 저장 및 SHAP Value 저장
      - SHAP : 각 테스트케이스 별 feature importance 를 SHAP Value 로 파악 가능
    - update() : 스트레스 예측값에 대한 유저의 평가를 받아 데이터 업데이트 & 모델 재학습
