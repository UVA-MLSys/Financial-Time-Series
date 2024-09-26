models=(DLinear PatchTST TimesNet iTransformer)
data_paths=('Apple.csv' 'Crude_Oil.csv' 'Exchange_Rate_Report.csv' 'Gold.csv' 'MSFT.csv' 'Natural_Gas.csv' 'SPX500.csv')
itrs=3
pred_len=24

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
    

    for model in ${models[@]}
    do 
    echo "Running for model:$model"

    python run.py \
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --model $model --itrs $itrs --disable_progress\
        --dump_output --itr_no 1

    done

    # MICN requires label_len to be equal to seq_len
    python run.py \
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --model MICN --disable_progress --itrs $itrs --label_len 96\
        --dump_output --itr_no 1

    python run.py\
        --model TimeMixer\
        --n_features $n_features --data_path $data_path\
        --down_sampling_layers 3 --down_sampling_window 2\
        --d_model 16 --d_ff 32 --label_len 0 \
        --down_sampling_method avg --e_layers 3 \
        --factor 3 --channel_independence 1 --itrs $itrs\
        --pred_len $pred_len --disable_progress \
        --dump_output --itr_no 1

    python run_CALF.py\
        --n_features $n_features --d_model 768\
        --data_path $data_path --disable_progress\
        --itrs $itrs --pred_len $pred_len\
        --model_id ori\
        --dump_output --itr_no 1

    python run_OFA.py\
        --n_features $n_features \
        --data_path $data_path --pred_len $pred_len\
        --itrs $itrs --d_model 768 --disable_progress\
        --model_id ori\
        --dump_output --itr_no 1

    python run_TimeLLM.py\
        --n_features $n_features --d_model 16\
        --data_path $data_path --pred_len $pred_len\
        --batch_size 16 --itrs $itrs --disable_progress\
        --model_id ori\
        --dump_output --itr_no 1

done 

data_path=Financial_Aid_State.csv
n_features=1
features=S
target=need_amt
seq_len=10
label_len=5
pred_len=1
itrs=3
top_k=2
patch_len=4
stride=2
freq=a
group_id=GROUP_ID

for model in ${models[@]}
do 
echo "Running for model:$model"

python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id $group_id\
    --model $model --itrs $itrs\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --top_k $top_k --freq $freq --target $target --dump_output --itr_no 1
done

# MICN requires label_len to be equal to seq_len
python run.py \
    --n_features $n_features --features $features\
    --data_path $data_path --group_id $group_id\
    --model MICN --disable_progress --itrs $itrs\
    --seq_len $seq_len --label_len $seq_len --pred_len $pred_len\
    --freq $freq --target $target --dump_output --itr_no 1

python run.py\
    --model TimeMixer\
    --n_features $n_features --data_path $data_path --group_id $group_id\
    --down_sampling_layers 3 --down_sampling_window 1\
    --d_model 16 --d_ff 32\
    --seq_len $seq_len --label_len 0 --pred_len $pred_len\
    --down_sampling_method avg --e_layers 3 --freq $freq\
    --factor 3 --channel_independence 1 --itrs $itrs\
    --features $features --target $target --dump_output --itr_no 1

python run_CALF.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id $group_id\
    --itrs $itrs\
    --model_id ori --freq $freq --d_model 768\
    --seq_len $seq_len --label_len $label_len\
    --pred_len $pred_len --target $target --dump_output --itr_no 1

python run_OFA.py\
    --n_features $n_features --features $features \
    --data_path $data_path --group_id $group_id\
    --itrs $itrs --d_model 768\
    --model_id ori --patch_size $patch_len --stride $stride\
    --seq_len $seq_len --label_len $label_len\
    --pred_len $pred_len --target $target --dump_output --itr_no 1

python run_TimeLLM.py\
    --n_features $n_features --d_model 16\
    --data_path $data_path --group_id $group_id --freq $freq \
    --batch_size 16 --itrs $itrs --disable_progress\
    --seq_len $seq_len --label_len $label_len --pred_len $pred_len\
    --model_id ori --top_k $top_k\
    --patch_len $patch_len --stride $stride --target $target --dump_output --itr_no 1