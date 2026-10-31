#!/bin/bash
# RSVQAxBEN test split (Zenodo 5084904, CC-BY-4.0): images, questions and answers JSON, 1.9 GB, MD5 checked.
# EarthJev uses RSVQAxBEN only as a transfer test, never for training.
#   bash scripts/data/download_rsvqaxben.sh [out dir]   (default data/rsvqaxben/raw)
set -euo pipefail
OUT="${1:-data/rsvqaxben/raw}"
mkdir -p "${OUT}"
get() {   # get <file> <md5>
  curl -sS -L --retry 5 -C - -o "${OUT}/$1" "https://zenodo.org/records/5084904/files/$1?download=1"
  echo "$2  ${OUT}/$1" | md5sum -c -
}
get RSVQAxBEN_split_test_images.json 73df473db7f7dd01fb96a9a4f3bfd04c
get RSVQAxBEN_split_test_questions.json 0ccc69b08253297896fb05362581dfd3
get RSVQAxBEN_split_test_answers.json cbf605f6ff30f8a624a7ff7ef2216cf1
