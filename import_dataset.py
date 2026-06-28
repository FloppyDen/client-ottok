import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)