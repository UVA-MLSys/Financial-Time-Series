#!/usr/bin/env bash
#SBATCH --job-name="fewshot_vary_financial_aid_state"
#SBATCH --output=outputs/fewshot_vary_financial_aid_state.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=32GB

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python run.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --model PatchTST --itrs 3 --seq_len 10 --label_len 5 --pred_len 1 --top_k 2 --freq a --target need_amt --percent 25 --channel_independence 0
python run.py --model TimeMixer --n_features 1 --data_path Financial_Aid_State.csv --group_id GROUP_ID --down_sampling_layers 3 --down_sampling_window 1 --d_model 16 --d_ff 32 --seq_len 10 --label_len 0 --pred_len 1 --down_sampling_method avg --e_layers 3 --freq a --factor 3 --channel_independence 0 --itrs 3 --features S --target need_amt --percent 25
python run_CALF.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --model_id ori --freq a --d_model 768 --seq_len 10 --label_len 5 --pred_len 1 --target need_amt --percent 25
python run_OFA.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --d_model 768 --model_id ori --patch_size 4 --stride 2 --seq_len 10 --label_len 5 --freq a --pred_len 1 --target need_amt --percent 25
python run_TimeLLM.py --n_features 1 --d_model 16 --data_path Financial_Aid_State.csv --group_id GROUP_ID --freq a --batch_size 16 --itrs 3 --disable_progress --seq_len 10 --label_len 5 --pred_len 1 --model_id ori --top_k 2 --patch_len 4 --stride 2 --target need_amt --percent 25
python run.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --model PatchTST --itrs 3 --seq_len 10 --label_len 5 --pred_len 1 --top_k 2 --freq a --target need_amt --percent 50 --channel_independence 0
python run.py --model TimeMixer --n_features 1 --data_path Financial_Aid_State.csv --group_id GROUP_ID --down_sampling_layers 3 --down_sampling_window 1 --d_model 16 --d_ff 32 --seq_len 10 --label_len 0 --pred_len 1 --down_sampling_method avg --e_layers 3 --freq a --factor 3 --channel_independence 0 --itrs 3 --features S --target need_amt --percent 50
python run_CALF.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --model_id ori --freq a --d_model 768 --seq_len 10 --label_len 5 --pred_len 1 --target need_amt --percent 50
python run_OFA.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --d_model 768 --model_id ori --patch_size 4 --stride 2 --seq_len 10 --label_len 5 --freq a --pred_len 1 --target need_amt --percent 50
python run_TimeLLM.py --n_features 1 --d_model 16 --data_path Financial_Aid_State.csv --group_id GROUP_ID --freq a --batch_size 16 --itrs 3 --disable_progress --seq_len 10 --label_len 5 --pred_len 1 --model_id ori --top_k 2 --patch_len 4 --stride 2 --target need_amt --percent 50
python run.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --model PatchTST --itrs 3 --seq_len 10 --label_len 5 --pred_len 1 --top_k 2 --freq a --target need_amt --percent 75 --channel_independence 0
python run.py --model TimeMixer --n_features 1 --data_path Financial_Aid_State.csv --group_id GROUP_ID --down_sampling_layers 3 --down_sampling_window 1 --d_model 16 --d_ff 32 --seq_len 10 --label_len 0 --pred_len 1 --down_sampling_method avg --e_layers 3 --freq a --factor 3 --channel_independence 0 --itrs 3 --features S --target need_amt --percent 75
python run_CALF.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --model_id ori --freq a --d_model 768 --seq_len 10 --label_len 5 --pred_len 1 --target need_amt --percent 75
python run_OFA.py --n_features 1 --features S --data_path Financial_Aid_State.csv --group_id GROUP_ID --itrs 3 --d_model 768 --model_id ori --patch_size 4 --stride 2 --seq_len 10 --label_len 5 --freq a --pred_len 1 --target need_amt --percent 75
python run_TimeLLM.py --n_features 1 --d_model 16 --data_path Financial_Aid_State.csv --group_id GROUP_ID --freq a --batch_size 16 --itrs 3 --disable_progress --seq_len 10 --label_len 5 --pred_len 1 --model_id ori --top_k 2 --patch_len 4 --stride 2 --target need_amt --percent 75
