# SageMaker AI에 최신 FLUX 컴퓨터 비전 모델 배포

이 노트북에서는 SageMaker AI 엔드포인트에 FLUX.1-dev-bnb-4bit 모델을 배포하는 방법을 보여드립니다.

S3의 모델과 커스텀 핸들러를 사용하여 HuggingFace 허브의 컴퓨터 비전 모델을 배포합니다.


## 1. 배포

이 섹션에서는 다음을 수행합니다:

* S3 위치에 있는 커스텀 핸들러와 함께 `FLUX.1-dev-bnb-4bit` 모델을 배포하는 방법을 확인합니다.