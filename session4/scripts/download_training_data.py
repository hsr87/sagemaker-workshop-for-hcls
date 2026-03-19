#!/usr/bin/env python3
"""
BoltzGen 학습 데이터를 다운로드하는 스크립트

Boltz 프로젝트에서 공개한 RCSB 학습 데이터를 로컬 또는 SageMaker Processing Job에서
다운로드하여 S3에 업로드할 수 있도록 준비합니다.

데이터 구조:
  - targets/: RCSB 처리된 타겟 구조 데이터
  - msa/: 다중 서열 정렬(MSA) 데이터
  - mols/: 화합물(CCD) 데이터
"""

import os
import subprocess
import sys
from pathlib import Path


def download_file(url: str, output_path: Path):
    """wget을 사용하여 파일을 다운로드합니다."""
    print(f"Downloading {url} to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(output_path), url],
        check=True
    )
    print(f"Downloaded: {output_path} ({output_path.stat().st_size / 1e9:.2f} GB)")


def extract_tar(tar_path: Path, output_dir: Path):
    """tar 파일을 압축 해제합니다."""
    print(f"Extracting {tar_path} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xf", str(tar_path), "-C", str(output_dir)],
        check=True
    )
    print(f"Extracted to: {output_dir}")


def main():
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path("/tmp/downloads")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Boltz 공개 학습 데이터 URL
    data_urls = {
        "rcsb_targets": "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_targets.tar",
        "rcsb_msa": "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar",
        "ccd": "https://boltz1.s3.us-east-2.amazonaws.com/ccd.pkl",
        "symmetry": "https://boltz1.s3.us-east-2.amazonaws.com/symmetry.pkl",
    }

    download_targets = os.environ.get("DOWNLOAD_TARGETS", "rcsb_targets,ccd").split(",")

    print(f"Will download: {download_targets}")
    print(f"Output directory: {output_dir}")

    for name in download_targets:
        name = name.strip()
        if name not in data_urls:
            print(f"Unknown target: {name}, skipping...")
            continue

        url = data_urls[name]

        if url.endswith(".tar"):
            tar_path = tmp_dir / f"{name}.tar"
            download_file(url, tar_path)

            if "targets" in name:
                extract_dir = output_dir / "targets"
            elif "msa" in name:
                extract_dir = output_dir / "msa"
            else:
                extract_dir = output_dir / name

            extract_tar(tar_path, extract_dir)
            tar_path.unlink()
        else:
            if "ccd" in name:
                output_path = output_dir / "mols" / "ccd.pkl"
            else:
                output_path = output_dir / Path(url).name

            download_file(url, output_path)

    print("\n" + "=" * 60)
    print("Download complete! Output contents:")
    print("=" * 60)
    for item in output_dir.rglob("*"):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.relative_to(output_dir)}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
