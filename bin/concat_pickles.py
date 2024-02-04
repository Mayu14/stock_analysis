#!/usr/bin/env python3
import sys
import datetime
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback


if __name__ == '__main__':
    pickle_file_base  = 'stock_check'

    p = Path(r'./').glob(f'{pickle_file_base}_*.pickle')
    files = [ x for x in p if x.is_file() ]
    dfs = []
    for fname in files:
        with open(fname, 'rb') as f:
            dfs.append( pickle.load(f) )

    df = pd.concat( dfs, ignore_index=False )

    print(df)
    df.to_csv('concat.csv')
    df.to_pickle('concat.pickle.xz')





