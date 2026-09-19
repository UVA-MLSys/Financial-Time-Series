#!/usr/bin/env bash
#SBATCH --job-name="fewshot_Financial_Aid"
#SBATCH --output=outputs/fewshot_Financial_Aid.out
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=32GB

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python run.py --n_features 4 --features MS --data_path Financial_Aid.csv --group_id GROUP_ID --model PatchTST --itrs 3 --disable_progress --seq_len 5 --label_len 3 --pred_len 1 --target OFFER_BALANCE --percent 10 --top_k 2 --freq a
python run.py --model TimeMixer --n_features 4 --data_path Financial_Aid.csv --group_id GROUP_ID --down_sampling_layers 3 --down_sampling_window 2 --d_model 16 --d_ff 32 --seq_len 5 --label_len 0 --pred_len 1 --down_sampling_method avg --e_layers 3 --freq a --factor 3 --channel_independence 1 --itrs 3 --features MS --target OFFER_BALANCE --disable_progress --percent 10
python run_CALF.py --n_features 4 --features MS --data_path Financial_Aid.csv --group_id GROUP_ID --itrs 3 --disable_progress --model_id ori --freq a --d_model 768 --seq_len 5 --label_len 3 --pred_len 1 --target OFFER_BALANCE --percent 10
python run_OFA.py --n_features 4 --features MS --data_path Financial_Aid.csv --group_id GROUP_ID --itrs 3 --disable_progress --model_id ori --d_model 768 --freq a --seq_len 5 --label_len 3 --pred_len 1 --target OFFER_BALANCE --percent 10 --patch_size 4 --stride 2
python run_TimeLLM.py --n_features 4 --d_model 16 --data_path Financial_Aid.csv --group_id GROUP_ID --freq a --batch_size 16 --itrs 3 --disable_progress --seq_len 5 --label_len 3 --pred_len 1 --model_id ori --percent 10 --target OFFER_BALANCE --top_k 2 --patch_len 4 --stride 2
