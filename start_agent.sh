#!/bin/bash

cd /Users/dann24/stock-ai-agent

# Raise file descriptor limit — needed for broad market scanning (yfinance + SQLite + HTTP)
ulimit -n 4096

source /opt/anaconda3/etc/profile.d/conda.sh

conda activate base

python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN
