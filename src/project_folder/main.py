import duckdb as dd
import os
import sys

from numpy import require, shape, mean, sqrt
import pandas as pd
from sklearn.metrics import mean_squared_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import argparse
from rich import print as rprint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scripts import data_loader as dl, embeddings as em
from config import data_loader_conf, embeddings_conf, train_conf
from loguru import logger
from scripts import train_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import streamlit as st

dd.execute("INSTALL motherduck; LOAD motherduck;")
# duckdb.execute(f"SET motherduck_token=current_setting('MOTHERDUCK_TOKEN');")

ac_tkn = st.secrets["MOTHERDUCK_TOKEN"] if st.secrets["MOTHERDUCK_TOKEN"] else os.getenv("MOTHERDUCK_TOKEN")
if ac_tkn is None:
    logger.error("MOTHERDUCK_TOKEN is not None")
else:
    dd.execute(f"SET motherduck_token='{ac_tkn}';")



def ftr_selection_processing(prediction_type, **kwargs):
    """
    Scale and apply PCA to image data for training, validation, and test sets,
    or a single batch of images.

    If `prediction_type` is 'batch', fits or loads a scaler + PCA pipeline,
    transforms train/val/test image sets, and returns them.

    Otherwise, loads the saved pipeline and transforms a single image batch.

    Parameters
    ----------
    prediction_type : str
        'batch' for train/val/test processing; otherwise, single batch.
    **kwargs :
        - If 'batch': 'train', 'val', 'test' DataFrames (image features).
        - Else: 'data' DataFrame (image features).

    Returns
    -------
    tuple or ndarray
        Transformed image data.
    """

    if prediction_type == 'batch':
        train = kwargs.get('train')
        val = kwargs.get('val')
        test = kwargs.get('test')
        
        # print(train.head(2))
        if 'scaler_pca_pipeline.pkl' not in os.listdir('models/'):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=50))
            ])

            pipeline.fit(train)
            train_pca = pipeline.transform(train)
            joblib.dump(pipeline, 'models/scaler_pca_pipeline.pkl')
            val_pca = pipeline.transform(val)
            test_pca = pipeline.transform(test) 

        else:
            pipeline = joblib.load('models/scaler_pca_pipeline.pkl')
            train_pca = pipeline.transform(train)
            val_pca = pipeline.transform(val)
            test_pca = pipeline.transform(test)
        return train_pca, val_pca, test_pca
    else:
        data = kwargs.get('data')
        pipeline = joblib.load('models/scaler_pca_pipeline.pkl')
        data = pipeline.transform(data)
        return data


def train_and_score(data, target, prediction_type):
    if prediction_type == 'batch':
        tr_conf = train_conf.train_val_test
        train_df, val_df = train_test_split(data, test_size = tr_conf.get('train_val_ratio'), shuffle = True, stratify = data[tr_conf.get('stratify')].values, random_state = tr_conf.get('random_state'))                            
        val_df, test_df = train_test_split(val_df, test_size = tr_conf.get('val_test_ratio'), shuffle = True, random_state = tr_conf.get('random_state'))                                                            
        logger.info(f'train_df.shape : {train_df.shape} val_df.shape : {val_df.shape} test_df.shape : {test_df.shape}')                                                                                                                
                                                                                                                                                                                
        x_train = train_df.drop(tr_conf.get('target_cols'), axis = 1)
        x_val = val_df.drop(tr_conf.get('target_cols'), axis = 1)
        x_test = test_df.drop(tr_conf.get('target_cols'), axis = 1)

        x_train_pca, x_val_pca, x_test_pca = ftr_selection_processing(prediction_type = prediction_type, train = x_train, val = x_val, test = x_test)

        y_train = train_df[target].astype(float) if target == 'age' else train_df[target]
        y_val = val_df[target].astype(float) if target == 'age' else val_df[target]
        y_test = test_df[target].astype(float) if target == 'age' else test_df[target]
        
        logger.info('Training started ......')
        train_model.train(target, x_train_pca, x_val_pca, y_train,y_val)
        
        logger.info('Scoring started .......')
        
        return train_model.score(target, prediction_type, x_test = x_test_pca, y_test = y_test)
    
    elif prediction_type == 'single':
       return train_model.score(target, prediction_type, data = data)

    else:
         return 'select either "single" or "batch" in prediction_type'



def main(prediction_type, target, **kwargs):
    embedding_needed = kwargs.get('embedding_needed')
    print("Hello from age-gender-race-project!")
    if prediction_type == 'batch': 
        df = dl.apply_filter(dl.data_load_df(dl.data_mount(data_loader_conf)))
        conn = dd.connect(embeddings_conf.db_name)
        conn.execute('create table if not exists raw_filtered_data as select * from df')
        if embedding_needed == True:
            for sub_dir in os.listdir('data/'):
                if sub_dir.startswith('sub_dir'):
                    logger.info(f"Creating embeddings for the folder : {sub_dir}")
                    em.batch_embeddings(sub_dir, embeddings_conf)
            conn.execute('''
                create table if not exists filtered_embedding_table as select * from 
                (select a.image_name, a.age, a.gender, a.race, a.age_interval, b.* from raw_filtered_data as a inner join feature_table as b on a.image_name=b.images);
                alter table filtered_embedding_table drop column image_name;
                             ''')
        data = conn.sql("select * from filtered_embedding_table").df().drop(['images'], axis = 1)
        train_and_score(data = data, target = target, prediction_type = prediction_type)
        logger.info('training and scoring is completed')

    else:
        image_path = kwargs.get('image_path')
        df = pd.DataFrame(em.get_embedding(image_path)).rename(columns = {i:f'ftr_{i+1}' for i in range(512)})
        data = ftr_selection_processing(prediction_type, data = df)
        y_pred = train_and_score(data, target, prediction_type).astype(str)[0]
        pred = train_conf.target_mapping[target][y_pred] if target != 'age' else round(float(y_pred))

        logger.info(f"so the final prediction for {target} is : {pred}")
        return pred



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_type', type = str, required = True, help = 'single or batch')
    parser.add_argument('--target', type = str, required = True, help = 'Gender, Race or Age')
    parser.add_argument('--embedding_needed', action = "store_true", required = False, help = 'if feature table is present, it should be false')
    parser.add_argument('--image_path', type = str, required = False, help = 'image_path')

    args = parser.parse_args()
    image_path = args.image_path if args.image_path else ''


    main(args.prediction_type, args.target, embedding_needed = args.embedding_needed, image_path = image_path)
