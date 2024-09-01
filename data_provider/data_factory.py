from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

def data_provider(
    args, flag:str
) -> tuple[Dataset_Custom, DataLoader]:    
    shuffle_flag = flag == 'train'
    drop_last = False
    
    data_set = Dataset_Custom(
        args, flag=flag
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last
    )
    
    return data_set, data_loader
