#!/usr/bin/env python3
"""
BoltzGen Training Script for SageMaker Training Job

원본 저장소(https://github.com/HannesStark/boltzgen) 방식을 최대한 유지하면서
SageMaker Training Job 환경에 맞게 조정한 엔트리포인트입니다.
"""

import os
import sys
import subprocess
from pathlib import Path


def install_dependencies():
    """pip install boltzgen[dev] 설치."""
    print("Installing BoltzGen...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--quiet', 'boltzgen[dev]'
    ])
    # boltzgen CLI로 mol 데이터 다운로드 (캐시에 저장)
    print("Downloading moldir via boltzgen CLI...")
    subprocess.run([
        sys.executable, '-m', 'boltzgen.cli.boltzgen', 'download', 'moldir',
        '--cache', '/tmp/boltzgen_cache'
    ], capture_output=True)
    print("Installation complete.")


install_dependencies()

import json
import logging
import argparse
import shutil
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_boltzgen_config_path() -> Path:
    """BoltzGen 설정 디렉토리 경로를 탐색합니다."""
    try:
        import boltzgen
        return Path(boltzgen.__file__).parent / 'resources' / 'config' / 'train'
    except ImportError:
        return Path('/opt/ml/code/src/boltzgen/resources/config/train')


def parse_args():
    parser = argparse.ArgumentParser(description='BoltzGen Training on SageMaker')
    parser.add_argument('--config', type=str, default='boltzgen_small')
    parser.add_argument('--config-path', type=str, default=None)
    parser.add_argument('--name', type=str, default='boltzgen-sagemaker')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--gradient-accumulation', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every-n-steps', type=int, default=2500)
    parser.add_argument('--save-top-k', type=int, default=3)
    parser.add_argument('--tensorboard-dir', type=str, default=None)
    parser.add_argument('--disable-validation', action='store_true')
    parser.add_argument('--skip-pretrained', action='store_true')
    return parser.parse_args()


def get_sagemaker_env():
    env = {
        'model_dir': os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
        'tensorboard_dir': os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data') + '/tensorboard',
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
    if env['world_size'] > 1:
        os.environ['MASTER_ADDR'] = env['master_addr']
        os.environ['MASTER_PORT'] = env['master_port']
        os.environ['WORLD_SIZE'] = str(env['world_size'])
        os.environ['RANK'] = str(env['node_rank'] * env['num_gpus'])


def prepare_data_paths(env: dict) -> dict:
    """SageMaker 채널에서 데이터 경로를 준비합니다."""
    paths = {}
    training_dir = Path(env['training_dir'])
    checkpoints_dir = Path(env['checkpoints_dir'])

    if training_dir.exists():
        rcsb_targets = training_dir / 'targets' / 'rcsb_processed_targets'
        if rcsb_targets.exists():
            paths['target_dir'] = str(rcsb_targets)
        elif (training_dir / 'targets').exists():
            paths['target_dir'] = str(training_dir / 'targets')

        if (training_dir / 'msa').exists():
            paths['msa_dir'] = str(training_dir / 'msa')

        # mol 파일: boltzgen download로 받은 캐시 사용, 없으면 번들 파일 복사
        mols_dir = training_dir / 'mols'
        mols_dir.mkdir(parents=True, exist_ok=True)
        if not (mols_dir / 'ALA.pkl').exists():
            # boltzgen download로 받은 mol 캐시 확인
            cache_mols = Path('/tmp/boltzgen_cache/boltzgen/mols')
            if not cache_mols.exists():
                cache_mols = Path(os.path.expanduser('~/.cache/boltzgen/mols'))
            if cache_mols.exists():
                logger.info(f"Copying mol files from boltzgen cache: {cache_mols}")
                for pkl in cache_mols.glob('*.pkl'):
                    shutil.copy2(pkl, mols_dir / pkl.name)
            else:
                # fallback: source_dir에 번들된 mol 파일 (바이너리 복사)
                bundled = Path('/opt/ml/code/data/mols')
                if bundled.exists():
                    logger.info(f"Copying bundled mol files from {bundled}")
                    for pkl in bundled.glob('*.pkl'):
                        shutil.copy2(pkl, mols_dir / pkl.name)
            logger.info(f"Mol files: {len(list(mols_dir.glob('*.pkl')))} files")
        paths['moldir'] = str(mols_dir)

        logger.info(f"Training directory contents: {list(training_dir.iterdir())[:10]}")

    if checkpoints_dir.exists():
        for ckpt_file in checkpoints_dir.glob('*.ckpt'):
            if 'pretrained' not in paths:
                paths['pretrained'] = str(ckpt_file)

    return paths


def build_config_overrides(args, env: dict, data_paths: dict) -> list:
    overrides = []

    overrides.append(f"output={env['model_dir']}")
    overrides.append(f"name={args.name}")
    overrides.append(f"trainer.devices={env['num_gpus']}")

    if 'target_dir' in data_paths:
        overrides.append(f"data.datasets.0.target_dir={data_paths['target_dir']}")
        overrides.append(f"data.datasets.0.manifest_path={data_paths['target_dir']}/manifest.json")
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
        if 'target_dir' in data_paths:
            overrides.append(f"data.monomer_target_dir={data_paths['target_dir']}")
            overrides.append(f"data.ligand_target_dir={data_paths['target_dir']}")
        overrides.append("data.datasets.0.filters=[]")

    if args.skip_pretrained:
        overrides.append("pretrained=null")

    return overrides


def run_training(args, env: dict, config_overrides: list):
    import hydra
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_path = Path(args.config_path) if args.config_path else get_boltzgen_config_path()

    if not config_path.exists():
        raise FileNotFoundError(f"Config path not found: {config_path}")

    logger.info(f"Config: {config_path}/{args.config}")
    logger.info(f"Overrides: {config_overrides}")

    with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
        cfg = compose(config_name=args.config, overrides=config_overrides)
        logger.info(OmegaConf.to_yaml(cfg))
        training_task = hydra.utils.instantiate(cfg)
        training_task.run(cfg)


def save_final_model(env: dict):
    model_dir = Path(env['model_dir'])
    checkpoints = list(model_dir.glob('**/*.ckpt'))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
        final_path = model_dir / 'model.ckpt'
        if latest_ckpt != final_path:
            shutil.copy2(latest_ckpt, final_path)
        print(f"[Metric] final_model_size_mb={final_path.stat().st_size / (1024*1024):.2f}")
    else:
        logger.warning("No checkpoints found")


def main():
    logger.info("=" * 60)
    logger.info("BoltzGen Training on SageMaker")
    logger.info("=" * 60)

    args = parse_args()
    env = get_sagemaker_env()
    for k, v in env.items():
        logger.info(f"  {k}: {v}")

    setup_distributed(env)

    tensorboard_dir = args.tensorboard_dir or env['tensorboard_dir']
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

    data_paths = prepare_data_paths(env)
    logger.info(f"Data paths: {data_paths}")

    config_overrides = build_config_overrides(args, env, data_paths)
    print("[Metric] training_started=1")

    try:
        run_training(args, env, config_overrides)
        logger.info("Training completed successfully!")
        print("[Metric] training_completed=1")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print("[Metric] training_failed=1")
        raise
    finally:
        save_final_model(env)

    logger.info("Training job finished")


if __name__ == "__main__":
    main()
