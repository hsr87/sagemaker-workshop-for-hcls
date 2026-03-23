# SageMaker AI에 딥러닝 모델 배포
---
이 노트북에서는 다양한 컨테이너 옵션을 사용하여 딥러닝 모델을 SageMaker AI 엔드포인트에 배포하는 방법을 보여드립니다.

간단한 옵션(또는 JumpStart나 Hugging Face 모델에서 배포)부터 시작하여, 더 고급 사용 사례 — S3에서 PyTorch 컨테이너로 모델을 배포하거나 NVIDIA의 특화된 Triton 컨테이너를 사용하는 방법까지 진행합니다.

## SageMaker 딥러닝 모델 배포 옵션

Amazon SageMaker는 다양한 사용 사례와 요구 사항에 최적화된 여러 배포 옵션을 제공합니다.

### 1. SageMaker Deep Learning Containers (DLCs)

AWS에서 관리하는 사전 구축된 컨테이너로, 인기 있는 ML 프레임워크와 최적화가 포함되어 있습니다.

**장점:**
-  최적화된 프레임워크(PyTorch, TensorFlow 등)로 사전 구성됨
-  정기적인 보안 업데이트 및 패치 제공
-  내장된 성능 최적화
-  SageMaker 기능과의 쉬운 통합

**사용 사례:**
-  표준 모델 배포
-  빠른 프로토타이핑
-  일반적인 프레임워크를 사용하는 모델

### 2. 특화 컨테이너

특정 추론 엔진과 최적화를 위해 설계된 맞춤형 컨테이너입니다.

#### NVIDIA Triton Inference Server
-  다중 프레임워크 지원(TensorFlow, PyTorch, ONNX, TensorRT)
-  동적 배칭(dynamic batching) 및 모델 앙상블(model ensembling)
-  GPU 메모리 최적화
-  동시 모델 실행

#### 기타 특화 옵션
-  **DJL Serving**: 고급 배칭 기능을 제공하는 Java 기반 서빙
-  **Text Generation Inference (TGI)**: LLM 추론에 최적화
-  **TensorRT**: NVIDIA의 고성능 추론 최적화기
-  **커스텀 컨테이너**: 특정 모델 아키텍처에 맞게 제작됨

**장점:**
-  특정 워크로드에 대한 전문화된 최적화
-  동적 배칭 같은 고급 기능
-  더 나은 리소스 활용
-  프레임워크별 최적화

### 올바른 옵션 선택하기

-  일반적인 프레임워크를 사용하는 표준 배포에는 **DLCs**를 사용하십시오
-  성능이 중요한 애플리케이션이나 특정 최적화가 필요한 경우 **특화 컨테이너**를 사용하십시오
-  배포 옵션을 선택할 때 모델 크기, 지연시간(latency) 요구사항, 처리량(throughput) 요구를 고려하십시오

## LAB1

### 1. SageMaker 네이티브 Deep Learning Containers(DCSs)에 모델 배포

이 섹션에서는 이미지 분류 모델을 세 가지 다른 옵션으로 SageMaker에 배포합니다:
-  SageMaker JumpStart에서 `ic-efficientnet-v2-imagenet21k-ft1k-m`을 SageMaker Python SDK로 배포
-  Hugging Face의 `microsoft/resnet-50`을 SageMaker 전용 컨테이너로 SageMaker Python SDK를 사용해 배포
-  pytorch.org의 `resnet-50` 모델. 사전 학습된 가중치 바이너리로 모델 아티팩트를 생성하고 사용자 정의 추론 스크립트를 작성한 후, 해당 아티팩트를 PyTorch 컨테이너에 배포합니다.

### 2. NVIDIA Triton 컨테이너에 모델 배포

이 섹션에서는 NVIDIA Triton Inference Server와 통합된 SageMaker에 모델을 배포하는 방법을 보여드립니다.
-  Hugging Face의 PyTorch BERT 모델을 Triton Server에 배포하는 방법
-  모델을 NVIDIA TensorRT로 내보내 성능을 향상시키고 SageMaker에서 호스팅하는 방법