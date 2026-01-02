import numpy as np
import fireducks.pandas as pd
import duckdb as dd
from keras_facenet import FaceNet
import os
import sys                          
import gc 
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from loguru import logger  
from config import embeddings_conf as conf


embedder = FaceNet()                                                                                                            
                                                                                                                                                                                                                               
def get_embedding(image_path):                                                                                                  
    try :
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (200, 200))
        embd = embedder.embeddings([image])[0].reshape(1, -1)                                                                   
        return embd                                                                                                             
    except:                                                                                                                     
        logger.error(f'image embeddings failed :( for the image {image_path}')



def batch_embeddings(sub_folder, conf):
    db = conf.db_name
    feature_table = conf.table_name


    conn = dd.connect(db)
    schema = 'images VARCHAR PRIMARY KEY,' + ', '.join([f'ftr_{i} DOUBLE' for i in range(1,513)])
    conn.execute(f"create table if not exists {feature_table} ( {schema} )") #
    
    data_dir = conf.data_dir + sub_folder + '/'
    image_list = os.listdir(data_dir)

    data_list = []
    error_images = []
    for image in image_list:
        try:
            embd = get_embedding(data_dir + image)
            if len(embd) > 0:
                data_list.append([image] + list(embd[0]))
        except:
            error_images.append(data_dir + image)
            logger.error(data_dir + image)

    pdf = pd.DataFrame(data_list).rename(columns = {0:'images'} | {i:f'ftr_{i}' for i in range(1,513)})
    conn.execute(f"insert or replace into {feature_table} select * from pdf")
    del pdf, data_list
    
    db_size = conn.execute(f'select count(*) from {feature_table}').df()
    sample_df = conn.execute(f'select * from {feature_table} limit 5').df()

    logger.info(db_size)
    logger.info(sample_df)
    conn.close
    gc.collect()











