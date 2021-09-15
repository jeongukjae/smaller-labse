#!/bin/bash
set -ex

for lang in "fra" "spa" "deu" "cmn" "ara" "ita" "jpn" "kor" "nld" "pol" "por" "tha" "tur" "rus" ; do
    python evaluate_tatoeba.py --lang $lang $@
done
