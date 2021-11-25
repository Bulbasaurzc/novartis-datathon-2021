# Import required packages
import numpy as np
import pandas as pd
import os
import pathlib
from tqdm import tqdm as tqdm
import time
import pickle
from datetime import date
import json
import urllib.request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)


# Define global path variables
base_path =  os.path.join(os.getcwd(), '..')
raw_path = os.path.join(base_path, 'data', 'raw')
interim_path = os.path.join(base_path, 'data', 'interim')
processed_path = os.path.join(base_path, 'data', 'processed')
results_path = os.path.join(base_path, 'data', 'results')
models_path = os.path.join(base_path, 'models')