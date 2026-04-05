#!/usr/bin/env bash

# Usage: ./validate-submission.sh <ping_url> [repo_dir]
#
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)

PING_URL=${1:-}
REPO_DIR=${2:-.}
if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

echo "================================================="
echo "       OpenEnv Submission Validator"
echo "================================================="

echo -e "\n[1/3] Pinging HF Space Application..."
# Space goes to sleep or takes a bit to reply sometimes. We'll give it a 5-second timeout, but it just needs a 200/404/405 depending on if it hits the FastApi root correctly.
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X GET -m 10 "$PING_URL")

echo "Received HTTP $STATUS_CODE"
if [ "$STATUS_CODE" == "200" ] || [ "$STATUS_CODE" == "404" ] || [ "$STATUS_CODE" == "405" ]; then
  echo "✅ Space is reachable."
else
  echo "❌ Error: Space is not reachable. Is it running?"
  exit 1
fi

echo -e "\n[2/3] Checking Docker Build..."
# Determine docker build context
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  echo "❌ Could not find Dockerfile in root or server/ folder."
  exit 1
fi

if DOCKER_BUILDKIT=1 docker build -q "$DOCKER_CONTEXT" > /dev/null; then
  echo "✅ Docker image builds successfully."
else
  echo "❌ Docker build failed."
  exit 1
fi

echo -e "\n[3/3] Running OpenEnv Structural Validation..."
cd "$REPO_DIR" || exit 1
if openenv validate; then
  echo "✅ Structural validation passed."
else
  echo "❌ Structural validation failed."
  exit 1
fi

echo "================================================="
echo "🎉 Congratulations! Your submission is passing!"
echo "================================================="
