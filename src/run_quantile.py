import argparse
import warnings
import pandas as pd
from scipy.optimize import OptimizeWarning
from models import quantile
from prepare import PRIVATE

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repetitions',type=int,default=300)
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options detected')
        quantile(pd.read_csv(PRIVATE/'youth_weekdays.csv'),args.repetitions)
