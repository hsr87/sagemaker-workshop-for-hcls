# SageMaker로 쉽게, 대규모 AI 학습부터 BioFM까지

AWS SageMaker를 활용한 대규모 AI 모델 학습과 BioFM(Boltzgen) 실습 워크샵입니다.

## 세션 구성

| 세션 | 주제 | 설명 |
|------|------|------|
| [Session 1](./session1/) | SageMaker HyperPod on Slurm 기본 세팅 | HyperPod 클러스터 구성 및 Slurm 환경 세팅 |
| [Session 2](./session2/) | SageMaker HyperPod on Slurm 학습 | Slurm 기반 LLM 분산 학습 실습 |
| [Session 3](./session3/) | SageMaker Serving | LLM 서빙/추론 엔드포인트 배포 |
| [Session 4](./session4/) | SageMaker for BioFM | BoltzGen 학습/추론 및 TxGemma 약물 특성 예측 |

## Session 4: SageMaker for BioFM (Boltzgen & TxGemma)

Session 4에서는 SageMaker를 활용하여 생명과학(Life Sciences) 분야의 Foundation Model을 학습하고 서빙하는 방법을 실습합니다.

### BoltzGen

BoltzGen은 Diffusion 모델 기반의 단백질 바인더 설계 도구입니다. 타겟 단백질에 결합하는 단백질, 펩타이드, 나노바디 등을 설계할 수 있습니다.

```
Design Spec (YAML) → Design (Diffusion) → Inverse Folding → Folding (Boltz-2) → Analysis → Filtering → 최종 바인더
```

| 단계 | 설명 |
|------|------|
| Design | Diffusion 모델로 바인더 백본 구조 생성 |
| Inverse Folding | 백본에 맞는 아미노산 시퀀스 설계 |
| Folding | 설계된 복합체 구조 예측 (Boltz-2) |
| Analysis & Filtering | pLDDT, RMSD 메트릭 기반 최종 선별 |

### TxGemma

TxGemma는 Google이 개발한 치료제 개발(Therapeutics) 특화 언어 모델로, SMILES 문자열 기반의 약물 특성(BBB 투과성, 독성, 용해도 등)을 예측합니다.

### 실습 노트북

| 노트북 | 내용 | SageMaker 기능 |
|--------|------|----------------|
| [1_training_job.ipynb](./session4/1_training_job.ipynb) | BoltzGen 모델 Fine-tuning | Training Job |
| [2_inference_job.ipynb](./session4/2_inference_job.ipynb) | 단백질 바인더 설계(추론) | Processing Job |
| [3_txgemma_endpoint.ipynb](./session4/3_txgemma_endpoint.ipynb) | TxGemma 약물 특성 예측 | Real-time Endpoint |

## 사전 준비사항

- AWS 계정 및 SageMaker 접근 권한
- SageMaker Studio JupyterLab 또는 Code Editor 환경
- 기본적인 Python 및 Linux 명령어 이해

## 실습 환경

본 워크샵은 SageMaker Studio의 JupyterLab 또는 Code Editor에서 실습할 수 있습니다.
