#!/bin/bash
# EuroSAT-MS (Zenodo 7711810, MIT; 2.1 GB), used in one ablation. Needs internet. MD5 from the Zenodo API.
#   bash scripts/data/download_msi_sources.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/eurosat/ms/raw
get() {  # get <url> <out> <md5>
  [ -f "$2" ] && echo "$3  $2" | md5sum -c --quiet && { echo "ok (already there) $2"; return; }
  curl -sS -L --retry 5 -C - -o "$2" "$1"
  echo "$3  $2" | md5sum -c
}
get https://zenodo.org/api/records/7711810/files/EuroSAT_MS.zip/content \
    data/eurosat/ms/raw/EuroSAT_MS.zip 091174add3c8e680a49244acf185b9f0
du -sh data/eurosat/ms/raw
