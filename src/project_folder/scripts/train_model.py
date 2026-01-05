import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import duckdb as dd
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_absolute_percentage_error
from joblib import dump, load
import shutil
import subprocess as sb


from config import train_conf



def auto_classification(target, x_train, x_val, y_train, y_val):
    
    sb.run("pip install lazypredict", shell = True, text = True)
    import lazypredict

    if target != 'age':
        model = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _ = model.fit(x_train, x_val, y_train, y_val)
    else:
        model = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _ = model.fit(x_train, x_val, y_train, y_val)
    return models

def custom_metric(target, y, y_pred):
    if target != 'age':
        return accuracy_score(y, y_pred)
    else:
        return 1 - mean_absolute_percentage_error(y, y_pred)


def train(target, x_train, x_val, y_train, y_val):
    
    sb.run("pip install lightgbm catboost", shell = True, text = True)
    from lightgbm impot LGBMRegressor
    from catboost import CatBoostRegressor

    model_map = {'SVC': SVC, 'KNN': KNeighborsClassifier} if target != 'age' else {'KNN': KNeighborsRegressor, 'SVR': SVR, 'LGBM': LGBMRegressor, 'CatBoost': CatBoostRegressor}
    conf = train_conf.conf
    modelling_dir = train_conf.modelling_dir + target + '/'
    metric = train_conf.metric_dict[target]
    

    try:
        os.listdir(modelling_dir)
    except:
        sb.run(f"mkdir -p {modelling_dir}", shell = True, text = True)
   

    score_map = {target: 'neg_mean_squared_error' if target == 'age' else 'accuracy'}
    if  len([model_name for model_name in model_map if f'best_model_{model_name}_{target}.pkl' in os.listdir(modelling_dir)]) == len(model_map):
        return logger.info('skipping the training step as model is already trained in the models directory!!!')
    else:
        accuracy_dict = {}
        for model_name in model_map:
            logger.info(f'training started for {model_name}')
            model =  model_map[model_name]()
            param_dict = conf.get(model_name)
            random_search = RandomizedSearchCV(
                    model, param_distributions=param_dict, 
                    n_iter=20, scoring=score_map[target], cv=5, random_state=42, verbose=2, n_jobs=-1)
            random_search.fit(x_train, y_train)

            logger.info(f"Best Parameters: {random_search.best_params_}")
            logger.info(f"Best Score: {random_search.best_score_}")
            
            dump(random_search.best_estimator_, modelling_dir + f'best_model_{model_name}_{target}.pkl')
            
            best_model = random_search.best_estimator_
            y_val_pred = best_model.predict(x_val)
            
            logger.info(f'Validation {metric} : {custom_metric(target, y_val, y_val_pred)}')
            accuracy_dict[model_name] = custom_metric(target, y_val, y_val_pred)


        final_model_name = pd.DataFrame(accuracy_dict, index = ['accuracy']).T.sort_values(by = 'accuracy', ascending = False).head(1).index[0]
        shutil.copy(modelling_dir + f'best_model_{final_model_name}_{target}.pkl', modelling_dir + f'final_best_model_{final_model_name}_{target}.pkl')
        

        train_df = pd.DataFrame(np.hstack([x_train, y_train.values.reshape(-1,1)]))
        train_df[f'predict_{final_model_name}_{target}'] = load(modelling_dir + f'final_best_model_{final_model_name}_{target}.pkl').predict(x_train)

        val_df = pd.DataFrame(np.hstack([x_val, y_val.values.reshape(-1,1)]))
        val_df[f'predict_{final_model_name}_{target}'] = load(modelling_dir + f'final_best_model_{final_model_name}_{target}.pkl').predict(x_val)

        conn = dd.connect('data/age_gender.db')
        conn.execute(f'create table if not exists train_table_{target} as select * from train_df')
        conn.execute(f'create table if not exists val_table_{target} as select * from val_df')
        conn.close()
        logger.info('train_table and val_table has been created')



def score(target, prediction_type, **kwargs):
    modelling_dir = train_conf.modelling_dir + target + '/'

    if len([i for i in os.listdir(modelling_dir) if i.startswith('final_best_model')]) == 0:
        return 'model is not trained, please train the model first!!!'
    else:
        final_model_name = max([i for i in os.listdir(modelling_dir) if i.startswith('final_best_model')], key=lambda f: os.path.getmtime(os.path.join(modelling_dir, f)))
        model = load(modelling_dir + final_model_name)

        if prediction_type == 'single':
            data = kwargs.get('data')
            return model.predict(data)

        elif prediction_type == 'batch':
            x_test = kwargs.get('x_test')
            y_test = kwargs.get('y_test')
            y_pred = model.predict(x_test)

            test_df = pd.DataFrame(np.hstack([x_test, y_test.values.reshape(-1,1)]))
            test_df[f'predict_{final_model_name}_{target}'] = y_pred

            conn = dd.connect(train_conf.db_name)
            conn.execute(f'create table if not exists test_table_{target} as select * from test_df')
            conn.close()

            # return accuracy_score(y_test, y_pred)

        else:
            return 'select either "single" or "batch" in prediction_type'













