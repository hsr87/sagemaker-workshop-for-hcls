#!/bin/bash
# BoltzGen Docker 이미지를 빌드하고 ECR에 푸시하는 스크립트

set -e

REGION=${AWS_REGION:-us-east-1}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPOSITORY_NAME="boltzgen-sagemaker"
IMAGE_TAG=${IMAGE_TAG:-latest}

echo "=========================================="
echo "BoltzGen ECR Build and Push"
echo "=========================================="
echo "Region: $REGION"
echo "Account ID: $ACCOUNT_ID"
echo "Repository: $REPOSITORY_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "=========================================="

# Step 1: ECR 리포지토리 생성 (없는 경우)
echo "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names $REPOSITORY_NAME --region $REGION 2>/dev/null || \
    aws ecr create-repository \
        --repository-name $REPOSITORY_NAME \
        --region $REGION \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256

echo "ECR repository ready"

# Step 2: ECR 로그인
echo ""
echo "Step 2: Logging in to ECR..."
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "Logged in to ECR"

# Step 3: Docker 이미지 빌드
echo ""
echo "Step 3: Building Docker image..."
FULL_IMAGE_NAME="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:$IMAGE_TAG"

# session4 디렉토리 기준으로 빌드
cd "$(dirname "$0")/.."

docker build \
    -f Dockerfile.sagemaker \
    -t $REPOSITORY_NAME:$IMAGE_TAG \
    -t $FULL_IMAGE_NAME \
    .

echo "Docker image built"

# Step 4: ECR에 푸시
echo ""
echo "Step 4: Pushing image to ECR..."
docker push $FULL_IMAGE_NAME

echo "Image pushed to ECR"

# Step 5: 결과 출력
echo ""
echo "=========================================="
echo "Build and Push Complete!"
echo "=========================================="
echo "Image URI: $FULL_IMAGE_NAME"
echo ""
echo "이 Image URI를 노트북에서 사용하세요."
echo "=========================================="

echo $FULL_IMAGE_NAME > scripts/image_uri.txt
echo "Image URI saved to scripts/image_uri.txt"
