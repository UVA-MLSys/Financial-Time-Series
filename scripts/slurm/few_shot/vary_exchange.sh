#!/usr/bin/env bash
#SBATCH --job-name="fewshot_vary_exchange"
#SBATCH --output=outputs/fewshot_vary_exchange.out
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=32GB

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python run.py --n_features 7 --data_path Exchange_Rate_Report.csv --model PatchTST --itrs 3 --disable_progress --percent 25
python run.py --model TimeMixer --n_features 7 --data_path Exchange_Rate_Report.csv --down_sampling_layers 3 --down_sampling_window 2 --d_model 16 --d_ff 32 --label_len 0 --down_sampling_method avg --e_layers 3 --factor 3 --channel_independence 1 --itrs 3 --disable_progress --percent 25
python run_CALF.py --n_features 7 --d_model 768 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --model_id ori --percent 25
python run_OFA.py --n_features 7 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --d_model 768 --model_id ori --batch_size 16 --percent 25
python run_TimeLLM.py --n_features 7 --d_model 16 --data_path Exchange_Rate_Report.csv --batch_size 16 --itrs 3 --disable_progress --model_id ori --percent 25
python run.py --n_features 7 --data_path Exchange_Rate_Report.csv --model PatchTST --itrs 3 --disable_progress --percent 50
python run.py --model TimeMixer --n_features 7 --data_path Exchange_Rate_Report.csv --down_sampling_layers 3 --down_sampling_window 2 --d_model 16 --d_ff 32 --label_len 0 --down_sampling_method avg --e_layers 3 --factor 3 --channel_independence 1 --itrs 3 --disable_progress --percent 50
python run_CALF.py --n_features 7 --d_model 768 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --model_id ori --percent 50
python run_OFA.py --n_features 7 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --d_model 768 --model_id ori --batch_size 16 --percent 50
python run_TimeLLM.py --n_features 7 --d_model 16 --data_path Exchange_Rate_Report.csv --batch_size 16 --itrs 3 --disable_progress --model_id ori --percent 50
python run.py --n_features 7 --data_path Exchange_Rate_Report.csv --model PatchTST --itrs 3 --disable_progress --percent 75
python run.py --model TimeMixer --n_features 7 --data_path Exchange_Rate_Report.csv --down_sampling_layers 3 --down_sampling_window 2 --d_model 16 --d_ff 32 --label_len 0 --down_sampling_method avg --e_layers 3 --factor 3 --channel_independence 1 --itrs 3 --disable_progress --percent 75
python run_CALF.py --n_features 7 --d_model 768 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --model_id ori --percent 75
python run_OFA.py --n_features 7 --data_path Exchange_Rate_Report.csv --itrs 3 --disable_progress --d_model 768 --model_id ori --batch_size 16 --percent 75
python run_TimeLLM.py --n_features 7 --d_model 16 --data_path Exchange_Rate_Report.csv --batch_size 16 --itrs 3 --disable_progress --model_id ori --percent 75
