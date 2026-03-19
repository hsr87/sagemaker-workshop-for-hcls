#!/usr/bin/env python3
"""
SageMaker Processing Script for BoltzGen Inference

SageMaker Processing Job 컨테이너 내부에서 실행되는 스크립트입니다.
BoltzGen CLI를 호출하여 단백질 바인더 설계를 수행합니다.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run BoltzGen in SageMaker Processing')
    parser.add_argument('--design-spec', type=str, required=True,
                        help='설계 사양 YAML 파일 경로')
    parser.add_argument('--protocol', type=str, default='protein-anything',
                        choices=['protein-anything', 'peptide-anything',
                                'protein-small_molecule', 'nanobody-anything'],
                        help='설계 프로토콜')
    parser.add_argument('--num-designs', type=int, default=10,
                        help='생성할 설계 수')
    parser.add_argument('--budget', type=int, default=2,
                        help='최종 다양성 최적화 설계 수')
    parser.add_argument('--devices', type=int, default=1,
                        help='사용할 GPU 디바이스 수')

    args = parser.parse_args()

    # SageMaker Processing 경로
    input_path = Path('/opt/ml/processing/input')
    output_path = Path('/opt/ml/processing/output')
    cache_path = Path('/opt/ml/processing/cache')

    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Cache path: {cache_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # 설계 사양 파일 탐색
    design_spec_path = input_path / args.design_spec
    if not design_spec_path.exists():
        yaml_files = list(input_path.rglob('*.yaml'))
        if yaml_files:
            design_spec_path = yaml_files[0]
            logger.info(f"Using found design spec: {design_spec_path}")
        else:
            logger.error(f"Design spec not found: {design_spec_path}")
            sys.exit(1)

    logger.info(f"Design specification: {design_spec_path}")

    # BoltzGen CLI 명령어 구성
    cmd_parts = [
        'boltzgen', 'run',
        str(design_spec_path),
        '--output', str(output_path / 'results'),
        '--protocol', args.protocol,
        '--num_designs', str(args.num_designs),
        '--budget', str(args.budget),
        '--cache', str(cache_path),
        '--devices', str(args.devices)
    ]

    cmd = ' '.join(cmd_parts)
    logger.info(f"Running command: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        logger.info("BoltzGen output:")
        logger.info(result.stdout)

        metadata = {
            'design_spec': str(design_spec_path),
            'protocol': args.protocol,
            'num_designs': args.num_designs,
            'budget': args.budget,
            'status': 'completed'
        }

        with open(output_path / 'job_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("Processing job completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"BoltzGen failed with error: {e}")
        logger.error(f"Output: {e.stdout}")

        metadata = {
            'design_spec': str(design_spec_path),
            'protocol': args.protocol,
            'num_designs': args.num_designs,
            'budget': args.budget,
            'status': 'failed',
            'error': str(e)
        }

        with open(output_path / 'job_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        sys.exit(1)


if __name__ == '__main__':
    main()
