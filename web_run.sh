#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels
cd /home/craft8244/KWS_WEB
source /home/craft8244/KWS_WEB/.venv/bin/activate

WAV_FILE=$1
GENDER=$2
AGE=$3
MODELS_PATH=/home/craft8244/KWS_WEB/models_data
WORD_LIST=영화관,도서관,주차,영화,예약,어디,확인,시간,대출,자리
WORD_LIST_ELDER=어디,시간,진료,위치,메뉴,예약,선생님,지금,여기,알레르기
WORD_LABEL=$4
OUTPUT_PATH=$5
CMD_TRAIN="python3 -m web_run"

$CMD_TRAIN $1 $2 $3 $MODELS_PATH $WORD_LABEL $WORD_LIST $WORD_LIST_ELDER $OUTPUT_PATH
