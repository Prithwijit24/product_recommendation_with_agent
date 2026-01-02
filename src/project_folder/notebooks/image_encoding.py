import subprocess as sb
import numpy as np
import fireducks.pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from tensorflow import keras

from sklearn.model_selection import train_test_split
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress all warnings
warnings.filterwarnings("ignore")





import duckdb as dd
conn = dd.connect('age_gender.db')
schema = 'images VARCHAR,' + ', '.join([f'ftr_{i} DOUBLE' for i in range(1,513)])
conn.execute(f"create table if not exists feature_table ( {schema} )")


data_dir = input().strip() + '/'
def get_embedding(image_path):
    try :
        image = Image.open(image_path).convert('RGB').resize(160, 160)
        image = (image - 127.5)/127.5
        image = np.expand_dims(image, axis = 0)
        embd = embedder.embeddings(image)
        return embd.reshape(1, -1)
    except:
        pass

image_list = os.listdir(data_dir)
print(image_list[:10])
data_list = []
for image in image_list:
    embd = get_embedding(data_dir + image)
    try:
        data_list.append([image] + list(embd[0]))
    except:
        continue
pdf = pd.DataFrame(data_list).rename(columns = {0:'images'} | {i:f'ftr_{i}' for i in range(1,513)})
conn.execute("insert into feature_table select * from pdf")
del pdf, data_list
print(f'iteration {i} completed')
conn.close