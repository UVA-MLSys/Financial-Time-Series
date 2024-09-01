#!/usr/bin/env bash
#SBATCH --job-name="exchange_rate"
#SBATCH --output=outputs/exchange_rate.out
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=("DLinear" "MICN" "SegRNN" "iTransformer" "PatchTST" "TimeMixer" "TimesNet")

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features 7 \
    --data_path Exchange_Rate_Report.csv\
    --model $model --disable_progress

done