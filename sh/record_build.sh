#!/bin/bash

OUTFILE=$1  
# 写入时间 不覆盖
date > ${OUTFILE}
bash tool/run.sh >> ${OUTFILE}


#     当前        1层      目录
# find . -maxdepth 1 -type d 
