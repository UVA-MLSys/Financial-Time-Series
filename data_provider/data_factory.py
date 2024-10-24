from data_provider.data_loader import Dataset_Custom, MultiTimeSeries, MultiTimeSeriesMultiTarget
from torch.utils.data import DataLoader
from typing import Union

def data_provider(
    args, flag:str
) -> tuple[Union[Dataset_Custom, MultiTimeSeries], DataLoader]:    
    shuffle_flag = flag == 'train'
    drop_last = False
    
    if args.data_path == 'Financial_Aid_State.csv':
        dataset = MultiTimeSeriesMultiTarget(args, flag=flag)
    elif 'group_id' in args and args.group_id: 
        dataset = MultiTimeSeries(args, flag=flag)
    else:
        dataset = Dataset_Custom(args, flag=flag)
    
    print(flag, len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )
    
    return dataset, data_loader
