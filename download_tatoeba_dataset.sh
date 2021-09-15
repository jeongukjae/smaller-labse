#!/bin/bash

set -ex
mkdir -p data/tatoeba/v1

for lang in "fra" "spa" "deu" "cmn" "ara" "ita" "jpn" "kor" "nld" "pol" "por" "tha" "tur" "rus" ; do
    curl -L "https://raw.githubusercontent.com/facebookresearch/LASER/2aa9cf8242f1030282be23a9cfa906fd011c4b2d/data/tatoeba/v1/tatoeba.${lang}-eng.${lang}" \
        -o "./data/tatoeba/v1/tatoeba.${lang}-eng.${lang}"
    curl -L "https://raw.githubusercontent.com/facebookresearch/LASER/2aa9cf8242f1030282be23a9cfa906fd011c4b2d/data/tatoeba/v1/tatoeba.${lang}-eng.eng" \
        -o "./data/tatoeba/v1/tatoeba.${lang}-eng.eng"
done
