#!/bin/bash
# Fetch the three public Habermas Machine dataset files the Fig. 4C analysis reads into data/.
#
# Source: the public release accompanying Tessler et al. (2024), Science 386, adq2852 -- Google DeepMind's
# GCS bucket `habermas_machine`, prefix `datasets/` (gs://habermas_machine/datasets/, served read-only over
# https://storage.googleapis.com/). The fourth released file, hm_all_final_preference_rankings.parquet, is
# not used here. MD5s below are the bucket's ETags, checked against the copies this report was built from.
#
#   bash scripts/download_data.sh          # skips files already present with the right checksum
set -eu
cd "$(dirname "$0")/.."
BASE=https://storage.googleapis.com/habermas_machine/datasets
mkdir -p data
while read -r md5 name; do
  out="data/$name"
  if [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ]; then
    echo "have $name"
    continue
  fi
  echo "downloading $name ..."
  curl -fL --retry 3 -o "$out.part" "$BASE/$name"
  got=$(md5sum "$out.part" | cut -d' ' -f1)
  [ "$got" = "$md5" ] || { echo "checksum mismatch for $name: got $got, want $md5" >&2; exit 1; }
  mv "$out.part" "$out"
done <<'FILES'
22dddc10acd72ee975f77d0e3c4adc63 hm_all_candidate_comparisons.parquet
3890689576997c745a61accab950cb3d hm_all_position_statement_ratings.parquet
2bfc92da02a0021a422df00d4ae83fc5 hm_all_round_survey_responses.parquet
FILES
echo "DATA_OK $(date)"
