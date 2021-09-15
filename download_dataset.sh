#!/bin/bash

set -ex
mkdir -p data
for lang in "en" "fr" "es" "de" "zh" "ar" "zh_classical" "it" "ja" "ko" "nl" "pl" "pt" "th" "tr" "ru" ; do
    wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2735/${lang}.txt.gz -P data
    gunzip data/${lang}.txt.gz
done
