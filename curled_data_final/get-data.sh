#!/bin/bash
set -e

BASE_URL="https://auspice.broadinstitute.org/charon"
PREFIX="sars-cov-2/ma-delta/20211005/cluster-unique-usher"
REFERER="https://auspice.broadinstitute.org/sars-cov-2/ma-delta/20211005/cluster-unique-usher"

echo "Fetching available datasets..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getAvailable?prefix=$PREFIX" \
  --compressed -o available.json

echo "Fetching main dataset..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getDataset?prefix=$PREFIX" \
  --compressed -o dataset.json

echo "Fetching root sequence..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getDataset?prefix=$PREFIX&type=root-sequence" \
  --compressed -o root-sequence.fasta

echo "Fetching tip frequencies..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getDataset?prefix=$PREFIX&type=tip-frequencies" \
  --compressed -o tip-frequencies.json

echo "✅ All data fetched successfully."
