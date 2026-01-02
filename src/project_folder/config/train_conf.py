from scipy.stats import uniform
import numpy as np

db_name = 'md:age_gender_md'
train_val_test = {
        'train_val_ratio': 0.1,
        'val_test_ratio': 0.2,
        'shuffle': True,
        'stratify': 'race',
        'random_state': 30980,
        'target_cols': ['age', 'gender', 'race', 'age_interval'],
        }

modelling_dir = 'models/'
conf = {
        'SVC': {
             'C': uniform(0.1, 10),  # Regularization parameter
             'gamma': uniform(0.01, 1),  # Kernel coefficient
             'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
    },
    'KNN': {
        'n_neighbors' : list(range(5, 50))
        },

    'SVR': {
         'C': uniform(0.1, 10),  # Regularization parameter
         'gamma': uniform(0.01, 1),  # Kernel coefficient
         'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
        },
    
    'LGBM': {
        'num_leaves': np.arange(20, 200, 10),  # Number of leaves in one tree
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size for each iteration
        'n_estimators': np.arange(100, 2001, 200),  # Number of boosting iterations
        'max_depth': np.arange(3, 16, 1),  # Max depth of the tree
        'min_child_samples': np.arange(10, 100, 10),  # Minimum samples per child
        'subsample': [0.6, 0.8, 1.0],  # Fraction of samples for fitting trees
        'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features for each tree
        'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5, 1.0],  # L2 regularization
        'min_split_gain': [0, 0.01, 0.1],  # Minimum gain to split a node
        'max_bin': [63, 127, 255],  # Max number of bins used for discretizing features
        },

    'CatBoost': {
        'iterations': np.arange(500, 3001, 500),  # Number of boosting iterations
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate
        'depth': np.arange(4, 12, 1),  # Depth of trees
        'l2_leaf_reg': [1, 3, 5, 7, 9],  # L2 regularization coefficient
        'border_count': [32, 64, 128],  # Number of splits for categorical features
        'bagging_temperature': [0.0, 0.5, 1.0],  # Controls the randomness of the bagging
        'max_ctr_complexity': [1, 2, 3],  # Maximum categorical feature interaction complexity
        'thread_count': [4, 8, 16],  # Number of threads for training (adjust according to your system)
        }
    }


target_mapping = {
        'gender': {'0': 'Male', '1': 'Female'},
        'race': {'0': 'White', '1': 'Black', '2': 'Asian', '3': 'Indian'}
        }

metric_dict = {
        'gender': 'accuracy',
        'race': 'accuracy',
        'age': 'mape'
        }


