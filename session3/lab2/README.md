# SageMaker AI에서 최신 파운데이션 모델 (Foundation Models) 배포하기

이 노트북에서는 최신 파운데이션 모델을 SageMaker AI 엔드포인트에 배포하는 방법을 보여드립니다.

간단한 사용 사례(JumpStart 모델 배포)로 시작한 후, 더 발전된 사용 사례(S3에서 모델 배포, Inference Component 활성화 엔드포인트 사용 및 커스텀 컨테이너)를 살펴보겠습니다.


## 1. 배포(간단)

이 섹션에서는 다음을 수행합니다:
-  SageMaker Python SDK를 사용하여 SageMaker JumpStart에서 `Llama-3.2-3B-Instruct`를 배포합니다.
-  관리형 LMI 컨테이너를 사용하여 HuggingFace 허브의 `Qwen3-4B-Thinking-2507` 모델을 배포합니다.


## 2. 배포(고급)

이 섹션에서는 보다 고급 배포 시나리오를 사용하는 방법을 보여드립니다:
-  S3 위치에서 모델을 배포하고 컨테이너 시작 시 Python 라이브러리를 업데이트하는 방법
-  Inference Component가 활성화된 엔드포인트에 ***양자화(quantized)***된 모델을 배포하는 방법
-  CloudFormation을 사용하여 모델을 배포하는 방법
-  자체 커스텀 컨테이너를 사용하여 모델을 배포하는 방법