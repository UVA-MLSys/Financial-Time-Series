#!/usr/bin/env bash
#SBATCH --job-name="Financial_Aid"
#SBATCH --output=outputs/Financial_Aid_fewshot2.out
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=16GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

conda deactivate
conda activate ml

models=(DLinear PatchTST TimesNet iTransformer)
data_path=Financial_Aid.csv
n_features=4
features=MS
seq_len=5
label_len=3
pred_len=1
itrs=3
top_k=2
patch_len=4
stride=2

target=OFFER_BALANCE
percent=10

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path\
    --model $model --itrs $itrs --disable_progress\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target --percent $percent--top_k $top_k --freq a
done

# MICN requires label_len to be equal to seq_len
python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path\
    --model MICN --disable_progress --itrs $itrs\
    --seq_len $seq_len --label_len $seq_len --pred_len $pred_len\
    --target $target --percent $percent --freq a

python run.py\
    --model TimeMixer\
    --n_features $n_features --data_path $data_path\
    --down_sampling_layers 3 --down_sampling_window 2\
    --d_model 16 --d_ff 32\
    --seq_len $seq_len --label_len 0 --pred_len $pred_len\
    --down_sampling_method avg --e_layers 3  --freq a\
    --factor 3 --channel_independence 1 --itrs $itrs  --features $features\
    --target $target --disable_progress --percent $percent

python run_CALF.py\
    --n_features $n_features --features $features \
    --data_path $data_path\
    --itrs $itrs --disable_progress\
    --model_id ori --freq a --d_model 768\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target --percent $percent

python run_OFA.py\
    --n_features $n_features --features $features \
    --data_path $data_path\
    --itrs $itrs --disable_progress\
    --model_id ori --d_model 768 --freq a\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --target $target --percent $percent \
    --patch_size $patch_len --stride $stride

python run_TimeLLM.py\
    --n_features $n_features --d_model 16\
    --data_path $data_path  --freq a\
    --batch_size 16 --itrs $itrs --disable_progress\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --model_id ori --percent $percent --target $target\
    --top_k 2 --patch_len $patch_len --stride $stride