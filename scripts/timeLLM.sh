# #!/usr/bin/env bash
# #SBATCH --job-name="Apple"
# #SBATCH --output=outputs/Apple.out
# #SBATCH --partition=gpu
# #SBATCH --time=1:00:00
# #SBATCH --gres=gpu:1
# #---SBATCH --nodelist=lynx01
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load cuda-toolkit cudnn-8.9.5_cuda12.x anaconda3

# conda deactivate
# conda activate ml

# data_paths=('Apple.csv' 'Crude_Oil.csv' 'Exchange_Rate_Report.csv' 'Gold.csv' 'MSFT.csv' 'Natural_Gas.csv' 'SPX500.csv')
data_paths=('Gold.csv' 'MSFT.csv' 'Natural_Gas.csv' 'SPX500.csv')
itrs=3

for data_path in ${data_paths[@]}
do 
    if [ $data_path == 'Exchange_Rate_Report.csv' ]
    then 
        n_features=7
    elif [ $data_path == 'SPX500.csv' ]
    then 
        n_features=4
    else
        n_features=5
    fi

#     python run_TimeLLM.py\
#         --n_features $n_features --d_model 16\
#         --data_path $data_path --percent 0 --test\
#         --batch_size 16 --itrs $itrs\
#         --model_id ori  --llm_model LLAMA --result_path scratch --llm_dim 4096
        
    python run_TimeLLM.py\
        --n_features $n_features --d_model 16\
        --data_path $data_path --percent 10\
        --batch_size 16 --itrs $itrs\
        --model_id ori  --llm_model LLAMA --result_path scratch --llm_dim 4096
        
done 


# data_path=Financial_Aid_State.csv
# n_features=1
# features=S
# target=need_amt
# seq_len=10
# label_len=5
# pred_len=1
# top_k=2
# patch_len=4
# stride=2
# freq=a
# group_id=GROUP_ID

# python run_TimeLLM.py\
#     --n_features $n_features --d_model 16\
#     --data_path $data_path --group_id $group_id --freq $freq \
#     --batch_size 16 --itrs $itrs\
#     --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
#     --model_id ori --top_k $top_k\
#     --patch_len $patch_len --stride $stride --target $target\
#     --llm_model LLAMA --result_path scratch --llm_dim 4096\
#     --percent 0 --test
    
# python run_TimeLLM.py\
#     --n_features $n_features --d_model 16\
#     --data_path $data_path --group_id $group_id --freq $freq \
#     --batch_size 16 --itrs $itrs\
#     --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
#     --model_id ori --top_k $top_k\
#     --patch_len $patch_len --stride $stride --target $target\
#     --llm_model LLAMA --result_path scratch --llm_dim 4096\
#     --percent 10