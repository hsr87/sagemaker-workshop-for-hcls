#!/usr/bin/env python3
"""
BoltzGen Training Script for SageMaker Training Job

SageMaker Training Job의 엔트리포인트로 사용되는 스크립트입니다.
- SageMaker 환경 변수 파싱
- 분산 학습 설정
- 체크포인트 관리
- TensorBoard 및 CloudWatch 로깅
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """DLC에 포함되지 않은 BoltzGen 의존성 패키지를 설치합니다."""
    print("Installing BoltzGen dependencies...")
    # boltzgen 소스코드를 source_dir에서 직접 사용 (pip 버전 dtype 호환 문제 회피)
    # 의존성만 개별 설치
    deps = [
        'pytorch-lightning>=2.0.0', 'hydra-core>=1.3.0', 'omegaconf>=2.3.0',
        'einx', 'einops', 'biotite', 'gemmi==0.6.5', 'mashumaro', 'biopython',
        'pydssp', 'logomaker', 'hydride', 'edit_distance', 'rdkit', 'pandas',
        'pdbeccdutils', 'numba', 'matplotlib', 'huggingface_hub',
    ]
    for dep in deps:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', dep])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {dep}")
    print("Dependencies installation complete.")


install_dependencies()


def patch_boltzgen_manifest(training_manifest_path: str):
    """pip 패키지 내장 manifest를 학습 데이터 manifest로 교체합니다."""
    try:
        import boltzgen
        pkg_dir = Path(boltzgen.__file__).parent
        for pkg_manifest in pkg_dir.rglob('manifest.json'):
            import shutil as _shutil
            _shutil.copy2(training_manifest_path, pkg_manifest)
            print(f"Patched package manifest: {pkg_manifest} <- {training_manifest_path}")
    except Exception as e:
        print(f"Warning: Could not patch manifest: {e}")


import json
import logging
import argparse
import shutil
from typing import Optional

import torch

# BoltzGen 소스 경로 추가
sys.path.insert(0, '/opt/boltzgen/src')
sys.path.insert(0, '/opt/ml/code/src')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_boltzgen_config_path() -> Path:
    """BoltzGen 설정 디렉토리 경로를 탐색합니다."""
    possible_paths = [
        Path('/opt/ml/code/config/train'),
        Path('/opt/ml/code/src/boltzgen/resources/config/train'),
        Path('/opt/boltzgen/config/train'),
    ]

    try:
        import boltzgen
        package_dir = Path(boltzgen.__file__).parent
        possible_paths.insert(0, package_dir / 'resources' / 'config' / 'train')
    except ImportError:
        pass

    for path in possible_paths:
        if path.exists():
            return path

    return Path('/opt/ml/code/src/boltzgen/resources/config/train')


def parse_args():
    """커맨드라인 인자 및 SageMaker 환경 변수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='BoltzGen Training on SageMaker')

    parser.add_argument('--config', type=str, default='boltzgen_small',
                        help='학습 설정 이름 (yaml 확장자 제외)')
    parser.add_argument('--config-path', type=str, default=None,
                        help='설정 디렉토리 경로')
    parser.add_argument('--name', type=str, default='boltzgen-sagemaker',
                        help='실험 이름')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='사전 학습된 체크포인트 경로')
    parser.add_argument('--resume', type=str, default=None,
                        help='학습 재개 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=-1,
                        help='최대 에폭 수 (-1: 무한)')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='최대 학습 스텝 수')
    parser.add_argument('--lr', type=float, default=None,
                        help='학습률')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='GPU당 배치 크기')
    parser.add_argument('--gradient-accumulation', type=int, default=None,
                        help='Gradient Accumulation 스텝 수')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader 워커 수')
    parser.add_argument('--save-every-n-steps', type=int, default=2500,
                        help='체크포인트 저장 주기')
    parser.add_argument('--save-top-k', type=int, default=3,
                        help='유지할 상위 체크포인트 수')
    parser.add_argument('--tensorboard-dir', type=str, default=None,
                        help='TensorBoard 로그 디렉토리')
    parser.add_argument('--disable-validation', action='store_true',
                        help='검증 비활성화')
    parser.add_argument('--skip-pretrained', action='store_true',
                        help='사전 학습 체크포인트 로딩 건너뛰기')

    return parser.parse_args()


def get_sagemaker_env():
    """SageMaker 환경 설정을 가져옵니다."""
    env = {
        'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
        'tensorboard_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data') + '/tensorboard',
        'input_dir': os.environ.get('SM_INPUT_DIR', '/opt/ml/input'),
        'training_dir': os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'),
        'checkpoints_dir': os.environ.get('SM_CHANNEL_CHECKPOINTS', '/opt/ml/input/data/checkpoints'),
        'num_gpus': int(os.environ.get('SM_NUM_GPUS', torch.cuda.device_count() if torch.cuda.is_available() else 1)),
        'num_nodes': int(os.environ.get('SM_NUM_NODES', 1)),
        'current_host': os.environ.get('SM_CURRENT_HOST', 'localhost'),
        'hosts': json.loads(os.environ.get('SM_HOSTS', '["localhost"]')),
        'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
        'master_port': os.environ.get('MASTER_PORT', '7777'),
        'current_instance_type': os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'local'),
        'training_job_name': os.environ.get('TRAINING_JOB_NAME', 'local'),
    }

    env['world_size'] = env['num_nodes'] * env['num_gpus']
    try:
        env['node_rank'] = env['hosts'].index(env['current_host'])
    except ValueError:
        env['node_rank'] = 0

    return env


def setup_distributed(env: dict):
    """PyTorch 분산 학습 환경을 설정합니다."""
    if env['world_size'] > 1:
        os.environ['MASTER_ADDR'] = env['master_addr']
        os.environ['MASTER_PORT'] = env['master_port']
        os.environ['WORLD_SIZE'] = str(env['world_size'])
        os.environ['RANK'] = str(env['node_rank'] * env['num_gpus'])

        logger.info(f"Distributed training setup:")
        logger.info(f"  World size: {env['world_size']}")
        logger.info(f"  Num nodes: {env['num_nodes']}")
        logger.info(f"  Node rank: {env['node_rank']}")
        logger.info(f"  GPUs per node: {env['num_gpus']}")
        logger.info(f"  Master: {env['master_addr']}:{env['master_port']}")


def generate_manifest(target_dir: Path):
    """target_dir에 manifest.json이 없으면 구조 파일로부터 자동 생성합니다."""
    import numpy as np

    manifest_path = target_dir / 'manifest.json'
    if manifest_path.exists():
        logger.info(f"Manifest already exists: {manifest_path}")
        return

    structures_dir = target_dir / 'structures'
    if not structures_dir.exists():
        logger.warning(f"Structures directory not found: {structures_dir}")
        return

    npz_files = list(structures_dir.glob('*.npz'))
    logger.info(f"Generating manifest.json from {len(npz_files)} structure files...")

    records = []
    for npz_file in npz_files:
        pdb_id = npz_file.stem
        record = {
            "id": pdb_id,
            "structure": {"resolution": 2.0, "method": "X-RAY DIFFRACTION", "released": "2020-01-01"},
            "chains": [],
            "interfaces": [],
        }
        try:
            data = np.load(npz_file, allow_pickle=True)
            if "chain_id" in data:
                for i, cid in enumerate(np.unique(data["chain_id"])):
                    record["chains"].append({
                        "chain_id": int(cid) if isinstance(cid, (int, np.integer)) else i,
                        "chain_name": str(cid),
                        "mol_type": 0,
                        "cluster_id": f"{pdb_id}_{cid}",
                        "msa_id": f"{pdb_id}_{cid}",
                        "num_residues": int(np.sum(data["chain_id"] == cid)),
                        "valid": True,
                    })
            data.close()
        except Exception as e:
            logger.warning(f"Could not read {pdb_id}: {e}")
        records.append(record)

    with open(manifest_path, 'w') as f:
        json.dump({"records": records}, f, indent=2)
    logger.info(f"Generated manifest.json with {len(records)} records")


def ensure_mol_files(mols_dir: Path):
    """mols 디렉토리에 개별 분자 pkl 파일(ALA.pkl 등)이 있는지 확인하고, 없으면 source_dir에서 복사합니다."""
    if (mols_dir / "ALA.pkl").exists():
        logger.info(f"Mol pkl files already exist in {mols_dir}")
        return

    # source_dir(/opt/ml/code)에 번들된 mol 파일 복사
    bundled_mols = Path('/opt/ml/code/data/mols')
    if bundled_mols.exists():
        logger.info(f"Copying bundled mol files from {bundled_mols} to {mols_dir}...")
        for pkl_file in bundled_mols.glob('*.pkl'):
            shutil.copy2(pkl_file, mols_dir / pkl_file.name)
        logger.info(f"Copied {len(list(mols_dir.glob('*.pkl')))} mol files")
        return

    logger.error(f"Bundled mol files not found at {bundled_mols}")
    raise FileNotFoundError(f"Mol pkl files not found in {mols_dir} or {bundled_mols}")


def prepare_data_paths(env: dict) -> dict:
    """SageMaker 채널에서 데이터 경로를 준비합니다."""
    paths = {}

    training_dir = Path(env['training_dir'])
    checkpoints_dir = Path(env['checkpoints_dir'])

    if training_dir.exists():
        rcsb_targets = training_dir / 'targets' / 'rcsb_processed_targets'
        if rcsb_targets.exists():
            paths['target_dir'] = str(rcsb_targets)
            generate_manifest(rcsb_targets)
        elif (training_dir / 'targets').exists():
            paths['target_dir'] = str(training_dir / 'targets')
            generate_manifest(training_dir / 'targets')

        if (training_dir / 'msa').exists():
            paths['msa_dir'] = str(training_dir / 'msa')

        # moldir: source_dir에 번들된 mol 파일 사용 (참조 소스코드와 호환)
        mols_dir = training_dir / 'mols'
        mols_dir.mkdir(parents=True, exist_ok=True)
        if not (mols_dir / 'ALA.pkl').exists():
            bundled = Path('/opt/ml/code/data/mols')
            if bundled.exists():
                logger.info(f"Copying bundled mol files from {bundled}...")
                for pkl in bundled.glob('*.pkl'):
                    shutil.copy2(pkl, mols_dir / pkl.name)
                logger.info(f"Copied {len(list(mols_dir.glob('*.pkl')))} mol files")
        paths['moldir'] = str(mols_dir)

        logger.info(f"Training directory contents: {list(training_dir.iterdir())[:10]}")

    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob('*.ckpt'):
            if 'pretrained' not in paths:
                paths['pretrained'] = str(ckpt_file)

        fold_ckpt = checkpoints_dir / 'boltz2_fold.ckpt'
        if fold_ckpt.exists():
            paths['folding_checkpoint'] = str(fold_ckpt)

    return paths


def build_config_overrides(args, env: dict, data_paths: dict, tensorboard_dir: str) -> list:
    """Hydra 설정 오버라이드를 생성합니다."""
    overrides = []

    overrides.append(f"output={env['model_dir']}")
    overrides.append(f"name={args.name}")
    overrides.append(f"trainer.devices={env['num_gpus']}")

    if 'target_dir' in data_paths:
        overrides.append(f"data.datasets.0.target_dir={data_paths['target_dir']}")
        overrides.append(f"+data.datasets.0.manifest_path={data_paths['target_dir']}/manifest.json")
    if 'msa_dir' in data_paths:
        overrides.append(f"data.datasets.0.msa_dir={data_paths['msa_dir']}")
    if 'moldir' in data_paths:
        overrides.append(f"data.moldir={data_paths['moldir']}")

    if args.pretrained:
        overrides.append(f"pretrained={args.pretrained}")
    elif 'pretrained' in data_paths:
        overrides.append(f"pretrained={data_paths['pretrained']}")

    if args.resume:
        overrides.append(f"resume={args.resume}")

    if 'folding_checkpoint' in data_paths:
        overrides.append(f"model.refolding_validator.folding_checkpoint={data_paths['folding_checkpoint']}")

    if args.epochs != -1:
        overrides.append(f"trainer.max_epochs={args.epochs}")
    if args.max_steps:
        overrides.append(f"+trainer.max_steps={args.max_steps}")
    if args.lr:
        overrides.append(f"model.training_args.max_lr={args.lr}")
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    if args.gradient_accumulation:
        overrides.append(f"trainer.accumulate_grad_batches={args.gradient_accumulation}")

    overrides.append(f"data.num_workers={args.num_workers}")
    overrides.append(f"save_every_n_train_steps={args.save_every_n_steps}")
    overrides.append(f"save_top_k={args.save_top_k}")
    overrides.append("wandb=null")

    if args.disable_validation:
        overrides.append("model.validators=[]")
        overrides.append("model.refolding_validator=null")
        overrides.append("model.num_val_datasets=0")
        overrides.append("trainer.num_sanity_val_steps=0")
        overrides.append("data.datasets.0.split=null")
        overrides.append("data.monomer_split=null")
        overrides.append("data.ligand_split=null")
        # monomer/ligand target_dir도 학습 데이터 경로로 통일
        if 'target_dir' in data_paths:
            overrides.append(f"data.monomer_target_dir={data_paths['target_dir']}")
            overrides.append(f"data.ligand_target_dir={data_paths['target_dir']}")
        # 필터 비활성화 (소량 데이터에서 필터링 방지)
        overrides.append("data.datasets.0.filters=[]")

    if args.skip_pretrained:
        overrides.append("pretrained=null")

    return overrides


def patch_manifest_loader(training_manifest_path: str):
    """볼츠겐 Manifest.load를 패치하여 학습 데이터 manifest를 강제 사용합니다."""
    from boltzgen.data.data import Manifest

    original_load = Manifest.load.__func__

    @classmethod
    def patched_load(cls, path):
        forced_path = Path(training_manifest_path)
        logger.info(f"Manifest.load intercepted: {path} -> {forced_path}")
        return original_load(cls, forced_path)

    Manifest.load = patched_load
    logger.info(f"Manifest.load patched to always use: {training_manifest_path}")


def run_training(args, env: dict, config_overrides: list, tensorboard_dir: str):
    """BoltzGen 학습 파이프라인을 실행합니다."""
    import hydra
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        tb_logger = TensorBoardLogger(
            save_dir=tensorboard_dir,
            name=args.name,
            version='',
        )
    except ImportError:
        tb_logger = None

    GlobalHydra.instance().clear()

    config_path = Path(args.config_path) if args.config_path else get_boltzgen_config_path()

    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Config name: {args.config}")
    logger.info(f"Overrides: {config_overrides}")

    with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
        cfg = compose(config_name=args.config, overrides=config_overrides)
        logger.info("Configuration loaded:")
        logger.info(OmegaConf.to_yaml(cfg))

        training_task = hydra.utils.instantiate(cfg)
        training_task.run(cfg)


def save_final_model(env: dict):
    """최종 체크포인트를 SageMaker 모델 디렉토리에 저장합니다."""
    model_dir = Path(env['model_dir'])
    checkpoints = list(model_dir.glob('**/*.ckpt'))

    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        final_path = model_dir / 'model.ckpt'
        if latest_ckpt != final_path:
            shutil.copy2(latest_ckpt, final_path)
            logger.info(f"Final model saved to: {final_path}")

        model_size_mb = final_path.stat().st_size / (1024 * 1024)
        print(f"[Metric] final_model_size_mb={model_size_mb:.2f}")
    else:
        logger.warning("No checkpoints found to save as final model")


def save_training_info(env: dict, args):
    """재현성을 위해 학습 정보를 저장합니다."""
    info = {
        'config': args.config,
        'name': args.name,
        'instance_type': env['current_instance_type'],
        'num_gpus': env['num_gpus'],
        'num_nodes': env['num_nodes'],
        'training_job_name': env['training_job_name'],
    }

    info_path = Path(env['model_dir']) / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def main():
    logger.info("=" * 60)
    logger.info("BoltzGen Training on SageMaker")
    logger.info("=" * 60)

    args = parse_args()
    env = get_sagemaker_env()

    logger.info("SageMaker Environment:")
    for key, value in env.items():
        logger.info(f"  {key}: {value}")

    setup_distributed(env)

    tensorboard_dir = args.tensorboard_dir or env['tensorboard_dir']
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

    data_paths = prepare_data_paths(env)
    logger.info(f"Data paths: {data_paths}")

    # Manifest.load를 monkey-patch하여 학습 데이터 manifest만 사용
    if 'target_dir' in data_paths:
        manifest_file = Path(data_paths['target_dir']) / 'manifest.json'
        if manifest_file.exists():
            patch_boltzgen_manifest(str(manifest_file))
            patch_manifest_loader(str(manifest_file))

    config_overrides = build_config_overrides(args, env, data_paths, tensorboard_dir)

    print("[Metric] training_started=1")

    try:
        run_training(args, env, config_overrides, tensorboard_dir)
        logger.info("Training completed successfully!")
        print("[Metric] training_completed=1")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("[Metric] training_failed=1")
        raise
    finally:
        save_final_model(env)
        save_training_info(env, args)

    logger.info("=" * 60)
    logger.info("Training job finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
