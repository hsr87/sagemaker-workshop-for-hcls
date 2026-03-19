# Session 4: Boltzgen 학습 및 추론

## 개요

BoltzGen은 Diffusion 모델 기반의 단백질 바인더 설계 도구입니다. 타겟 단백질에 결합하는 단백질, 펩타이드, 나노바디 등을 설계할 수 있습니다.

이 세션에서는 Amazon SageMaker를 활용하여 BoltzGen 모델을 **학습(Training Job)** 하고, **추론(Processing Job)** 하는 과정을 실습합니다.

## 학습 목표

- BoltzGen 모델의 아키텍처와 파이프라인 이해
- SageMaker Training Job을 활용한 모델 Fine-tuning
- SageMaker Processing Job을 활용한 단백질 바인더 설계(추론)
- 학습/추론 결과 모니터링 및 분석

## BoltzGen 파이프라인

```
Design (Diffusion) → Inverse Folding → Folding (Boltz-2) → Analysis → Filtering
```

| 단계 | 설명 | 리소스 |
|------|------|--------|
| Design | Diffusion 모델로 바인더 백본 구조 생성 | GPU |
| Inverse Folding | 백본에 맞는 아미노산 시퀀스 설계 | GPU |
| Folding | 설계된 복합체 구조 예측 (Boltz-2) | GPU |
| Analysis | pLDDT, RMSD 등 메트릭 계산 | CPU |
| Filtering | 품질 + 다양성 기반 최종 설계 선별 | CPU |

## 실습 구성

| 노트북 | 내용 | 소요 시간 |
|--------|------|-----------|
| [1_training_job.ipynb](./1_training_job.ipynb) | SageMaker Training Job으로 BoltzGen 모델 학습 | ~30분 |
| [2_inference_job.ipynb](./2_inference_job.ipynb) | SageMaker Processing Job으로 단백질 바인더 설계 | ~30분 |

## 사전 준비사항

- SageMaker Studio JupyterLab 또는 Code Editor 환경
- ECR에 BoltzGen Docker 이미지가 빌드되어 있어야 함
- S3 버킷 접근 권한이 포함된 SageMaker Execution Role

## 디렉토리 구조

```
session4/
├── README.md                     # 세션 개요 (현재 파일)
├── 1_training_job.ipynb          # 학습 실습 노트북
├── 2_inference_job.ipynb         # 추론 실습 노트북
├── scripts/
│   ├── train.py                  # Training Job 엔트리포인트
│   ├── processing_script.py      # Processing Job 엔트리포인트
│   ├── download_training_data.py # 학습 데이터 다운로드
│   └── build_and_push.sh         # Docker 이미지 빌드 스크립트
├── examples/
│   └── vanilla_protein.yaml      # 단백질 바인더 설계 사양 예제
├── Dockerfile.sagemaker          # SageMaker용 Docker 이미지
└── img/
```

## 인스턴스 권장 사양

### Training Job

| 인스턴스 | GPU | 메모리 | 용도 |
|----------|-----|--------|------|
| ml.g5.2xlarge | 1 | 32GB | 기본 Fine-tuning |
| ml.g5.12xlarge | 4 | 96GB | Multi-GPU 학습 |
| ml.p4d.24xlarge | 8 | 320GB | 대규모 학습 |

### Processing Job (추론)

| 인스턴스 | GPU | 메모리 | 용도 |
|----------|-----|--------|------|
| ml.g4dn.xlarge | 1 | 16GB | 소규모 테스트 |
| ml.g5.xlarge | 1 | 24GB | 일반 추론 |
| ml.g5.12xlarge | 4 | 96GB | 대규모 배치 추론 |
