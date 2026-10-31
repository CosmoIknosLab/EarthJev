#!/bin/bash
# Raw inputs of the CORINE and BigEarthNet.txt data (checksums verified):
#   BigEarthNet v2.0 (reBEN, Zenodo 10891137, CDLA-Permissive-1.0): metadata.parquet and Reference_Maps.tar.zst (0.29 GB).
#     The 63 GB Sentinel-2 archive is not downloaded: prepare_reben.py streams it and keeps only the selected patches.
#   BigEarthNet.txt (Hugging Face BIFOLD-BigEarthNetv2-0/BigEarthNet.txt, CDLA-Permissive-1.0): one parquet, 0.47 GB.
# RSVQAxBEN and EuroSAT-MS have their own scripts (download_rsvqaxben.sh, download_msi_sources.sh).
#   bash scripts/data/download_sources.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/reben/raw data/bentxt/raw
get() {  # get <url> <out> <md5>
  [ -f "$2" ] && echo "$3  $2" | md5sum -c --quiet && { echo "ok (already there) $2"; return; }
  curl -sS -L --retry 5 -C - -o "$2" "$1"
  echo "$3  $2" | md5sum -c
}
Z=https://zenodo.org/records/10891137/files
get "$Z/metadata.parquet?download=1" data/reben/raw/metadata.parquet 55687065e77b6d0b0f1ff604a6e7b49c
get "$Z/Reference_Maps.tar.zst?download=1" data/reben/raw/Reference_Maps.tar.zst 95d85a222fa983faddcac51a19f28917
F=data/bentxt/raw/BigEarthNet.txt.parquet
curl -sS -L --retry 5 -C - -o "$F" \
  "https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet"
echo "d3b97f999456016bb13c2a8e94b8f47825654f07a0394a6b266a38b750ca1554  $F" | sha256sum -c
