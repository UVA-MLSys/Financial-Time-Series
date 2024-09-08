data_paths=('Apple.csv' 'Crude_Oil.csv' 'Exchange_Rate_Report.csv' 'Gold.csv' 'MSFT.csv' 'Natural_Gas.csv' 'SPX500.csv')
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

    python run_CALF.py\
        --n_features $n_features --d_model 768\
        --data_path $data_path --disable_progress\
        --itrs $itrs\
        --model_id ori --zero_shot --test

    python run_OFA.py\
        --n_features $n_features \
        --data_path $data_path\
        --itrs $itrs --d_model 768 --disable_progress\
        --model_id ori --zero_shot --test

    python run_TimeLLM.py\
        --n_features $n_features --d_model 16\
        --data_path $data_path\
        --batch_size 16 --itrs $itrs --disable_progress\
        --model_id ori --zero_shot --test

done 
