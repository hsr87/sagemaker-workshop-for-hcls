# 최신 Open AI 자동 음성 인식(ASR) 모델을 SageMaker AI에 배포하기

이 노트북에서는 SageMaker AI 엔드포인트에 whisper-large-v2 모델을 배포하는 방법을 보여드립니다.

관리형 LMI 컨테이너를 사용해 HuggingFace 허브에서 ASR 모델을 배포한 다음, 보다 고급 사용 사례로 이동하여 S3에서 모델을 가져와 SageMaker HuggingFace Pytorch 컨테이너로 배포하는 방법을 보여드립니다.


## 1. 배포

이 섹션에서는 다음을 수행합니다:
-  관리형 LMI 컨테이너를 사용해 HuggingFace 허브에서 `whisper-large-v2` 모델을 배포합니다.
-  SageMaker HuggingFace Pytorch 컨테이너를 사용하여 S3 위치에서 `whisper-large-v2` 모델을 배포하는 방법을 확인합니다.
-  이를 사용하여 음성을 텍스트로 전사(transcribe)하고 한 언어에서 다른 언어로 번역(translate)하는 데 활용합니다.