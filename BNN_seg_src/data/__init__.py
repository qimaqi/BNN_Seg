import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as dset
import torchvision.transforms as transforms
from data.data_AMZ import AMZDataset
from data.data_camvid import CamVidDataset

def get_dataloader(config):
    data_dir = config.data_dir
    batch_size = config.batch_size
    test_batch_size = 4

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.0,), (1.0,))])

    if config.data_name == 'AMZ':
        val_percent = 0.1
        dataset = AMZDataset(data_dir,scale=0.25)
        n_val = int(len(dataset) * val_percent)
        n_train = len(dataset) - n_val
        train_dataset, test_dataset = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    elif config.data_name == 'CamVid':
        train_dataset = CamVidDataset(data_dir,usage='train',scale=0.5 )
        test_dataset = CamVidDataset(data_dir,usage='test',scale=0.5)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=config.num_work, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                             num_workers=config.num_work, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    data_loader = {'train': train_loader, 'test': test_loader}

    return data_loader
