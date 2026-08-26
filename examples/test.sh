#!/usr/bin/env bash

set -euo pipefail

# Paste the raw presigned_url value returned by create_presigned between the
# quotes. Do not include the surrounding JSON key or Markdown link formatting.
PRESIGNED_URL="http://localhost:8333/course-materials/3fa85f64-5717-4562-b3fc-2c963f66afa6/b1235848-980f-4b21-afa4-a5bda48c554c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=seaweedfs%2F20260826%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260826T083118Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=31813ce31ec7c2d5b9d7dee36039b46cba5da0ea8af236830150190744a75d82"
# Example:
# PRESIGNED_URL="http://localhost:8333/course-materials/<course-id>/<material-id>?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=...&X-Amz-Signature=..."

# Leave this empty to upload examples/sample-course-material.md, or set it to a
# different absolute or relative file path.
UPLOAD_FILE=""
# Example:
# UPLOAD_FILE="/Users/your-name/Documents/course-notes.md"

if [[ -z "${PRESIGNED_URL}" ]]; then
  echo "Set PRESIGNED_URL at the top of this script before running it." >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
file=${UPLOAD_FILE:-"${script_dir}/sample-course-material.md"}

if [[ ! -f "${file}" ]]; then
  echo "File not found: ${file}" >&2
  exit 1
fi

curl \
  --fail-with-body \
  --show-error \
  --request PUT \
  --upload-file "${file}" \
  "${PRESIGNED_URL}"

echo "Uploaded ${file} successfully."
