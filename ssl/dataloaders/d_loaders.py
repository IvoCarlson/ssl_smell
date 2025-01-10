'''
A script for loading the data and serving it to the model for pretraining
'''
import os,sys
sys.path.append(os.path.join(os.path.dirname("__file__"),'.'))

import numpy as np
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataloaders.malevis_dataset import MalevisDataset,rotnet_collate_fn,collate_noise
#from utils.transformations import TransformsSimCLR
import utils

def get_dataloaders(cfg,val_split=None):

    train_dataloaders,val_dataloaders,test_dataloaders = loaders(cfg)
    return train_dataloaders,val_dataloaders,test_dataloaders

def get_datasets(cfg):

    train_dataset,val_dataset,test_dataset = loaders(cfg,get_dataset=True,)
    return train_dataset,test_dataset

def loaders(cfg,get_dataset=False):

    if cfg.data_aug:
        data_aug = transforms.Compose([transforms.RandomResizedCrop(128)])

        if cfg.mean_norm == True:
            transform = transforms.Compose([data_aug,transforms.ToTensor(), transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])

        transform = transforms.Compose([data_aug,transforms.ToTensor()])

    elif cfg.mean_norm:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),transforms.ToTensor(),transforms.Normalize(mean=cfg.mean_pix, std=cfg.std_pix)])
    else:
        transform = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),transforms.ToTensor()])

    transform_test = transforms.Compose([transforms.Resize((cfg.img_sz,cfg.img_sz)),transforms.ToTensor()])

    if cfg.pretext=='rotation':
        collate_func=rotnet_collate_fn
    elif cfg.pretext=='noise' or cfg.pretext=='blur':
        collate_func=collate_noise
    else:
        collate_func=default_collate

    annotation_file = 'small_labeled_data.csv'

    train_dataset = MalevisDataset(cfg,annotation_file, data_type='train',transform=transform)
    val_dataset=None


    annotation_file = 'malevis_recognition_test.csv'

    test_dataset = MalevisDataset(cfg,annotation_file,data_type='test',transform=transform_test)
    if cfg.val_split:
        random_seed= 42
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(cfg.val_split * dataset_size))
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        dataloader_train = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  collate_fn=collate_func,sampler=train_sampler)

        dataloader_val = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                       collate_fn=collate_func,sampler=valid_sampler)

    else:
        dataloader_train = DataLoader(train_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)
        if val_dataset:
            dataloader_val = DataLoader(val_dataset,batch_size=cfg.batch_size,\
                                collate_fn=collate_func,shuffle=True)
        else:
            dataloader_val=None

    if get_dataset:

        return train_dataset,val_dataset,test_dataset

    dataloader_test = DataLoader(test_dataset,batch_size=cfg.batch_size,\
                            collate_fn=collate_func,shuffle=True)

    return dataloader_train,dataloader_val, dataloader_test
#%%
if __name__=='__main__':
    config_file=r'/home/ivo/data/SSL/ImpRotNet_09/ssl_pytorch/config/config_sl.yaml'
    cfg = utils.load_yaml(config_file,config_type='object')

#    tr_dset,ts_dset = get_datasets(cfg)

    tr_loaders,val_loaders,ts_loaders = get_dataloaders(cfg)

#    print ('length of tr_dset: {}'.format(len(tr_dset)))
#    print ('length of ts_dset: {}'.format(len(ts_dset)))


    data, label,idx,_ = next(iter(tr_loaders))
    print(data.shape, label)

    data, label,idx,_ = next(iter(ts_loaders))
    print(data.shape, label)
