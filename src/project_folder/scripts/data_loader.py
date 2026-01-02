import numpy as np
import fireducks.pandas as pd
import matplotlib.pyplot as plt
import subprocess as sb
import os
import sys
import gc
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
from config import data_loader_conf as conf

gc.enable()

def data_mount(conf):
    data_path = conf.data_mount_and_image_bucketing.get('data_path')
    
    if not data_path:
        data_mount_conf = conf.data_mount_and_image_bucketing['data_mount']
        password = data_mount_conf['password']
        is_mounted = sb.run(f'lsblk -o MOUNTPOINT,TYPE | grep {data_mount_conf["mount_point"]}', shell = True, text = True, capture_output=True)
        mount_command = f'sudo -S mount {data_mount_conf["mounted_drive"]} {data_mount_conf["mount_point"]}'
        unmount_command = f'sudo -S umount {data_mount_conf["mount_point"]}'
        if (is_mounted.stdout == ''):
                mounted_path = sb.run(mount_command, input=password + "\n", shell = True, text=True, capture_output=True)
                logger.info(f'data path has been mounted to {data_mount_conf["mount_point"]}!!!')
        else:
            logger.info(f'Already mounted, mountpoint --- {is_mounted.stdout}')
        data = os.listdir(data_mount_conf["data_path"])
    else:
        data_nested_list = [[os.path.join(data_path, folder, file) for file in os.listdir(os.path.join(data_path,folder))] for folder in os.listdir(data_path) if 'sub_dir' in folder]
        data = []
        for i in data_nested_list:
            data += i
    LS = os.listdir(data_path.split('data')[0])
    if ('data' not in LS):                                                                                                  
        sb.run(f"mkdir -p data", shell = True, text = True)
        image_bucket_conf = conf.data_mount_and_image_bucketing['image_bucketing']
        for i in tqdm(range(image_bucket_conf['no_of_buckets']), desc="image_bucketing", ascii=" >", bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            sb.run(f"mkdir -p data/sub_dir{i}", shell = True, text = True)
            for image in data[image_bucket_conf['batch_size']*i:image_bucket_conf['batch_size']*(i+1)]:
                sb.run("cp " + data_mount_conf['data_path'] + image + f" data/sub_dir{i}/", shell = True, text = True)
    else:
        logger.info('data already segregated into data folder. If data folder is empty remove the data folder and run the "data_mount" function once again')
    gc.collect()
    return data


def age_interval(x):
    if x/5 <= 20:
        for i in range(20):
            if (i*5) <= x <= ((i+1)*5):
                return f'{i*5 + 1} to {(i+1)*5}'
            else:
                continue
    else:
        return '100+'

def bw_check(x):
    return ((plt.imread(x)[:,:,0] == plt.imread(x)[:,:,1]).sum() == 40000) | \
    ((plt.imread(x)[:,:,0] == plt.imread(x)[:,:,2]).sum() == 40000) | \
    ((plt.imread(x)[:,:,1] == plt.imread(x)[:,:,2]).sum() == 40000)

def data_load_df(data):
    data_mount_conf = conf.data_mount_and_image_bucketing['data_path'] if conf.data_mount_and_image_bucketing['data_path'] is not None else conf.data_mount_and_image_bucketing['data_mount']
    data_df = pd.DataFrame({'images' : data})
    data_df['image_name'] = data_df['images'].apply(lambda row: row.split('/')[-1])
    data_df[['age', 'gender', 'race', 'photo']] = data_df['image_name'].str.split('_',expand = True)
   # data_df['images'] = data_df['images']
    data_df.drop('photo', axis = 1, inplace = True)
    data_df['age_interval'] = data_df['age'].astype(int).apply(age_interval)
    data_df['is_bw'] = data_df['images'].apply(lambda x: bw_check(x))
    gc.collect()
    return data_df

def apply_filter(data):
    data = data[data.race.isin(['0','1','2','3']) & (~data.age_interval.isin(['1 to 5'])) & (data.is_bw == False)]
    return data






















