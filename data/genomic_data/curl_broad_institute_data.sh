#!/bin/bash
set -e

BASE_URL="https://auspice.broadinstitute.org/charon"
PREFIX="sars-cov-2/ma-delta/20211005/cluster-unique-usher"
REFERER="https://auspice.broadinstitute.org/sars-cov-2/ma-delta/20211005/cluster-unique-usher"

echo "Fetching phylogenetic tree..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getDataset?prefix=$PREFIX" \
  --compressed -o tree.json

echo "Fetching root sequence..."
curl -L \
  -H "Referer: $REFERER" \
  "$BASE_URL/getDataset?prefix=$PREFIX&type=root-sequence" \
  --compressed -o root-sequence.json

echo "Data fetched successfully."