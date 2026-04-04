#!/bin/bash

cd /Users/dann24/stock-ai-agent

source /opt/anaconda3/etc/profile.d/conda.sh

conda activate base

python3 main.py --ticker BZAI AWRE LTRX BBAI SOUN --paper
