#!/usr/bin/env python3
import sys
import datetime
import yfinance as yf
import numpy as np
from scipy import integrate
import pandas as pd
import pickle
import time
import traceback

CONFIG_FILE = './etc/stock_symbol.txt'
VERBOSE = 0
VERSION = 0.1

DAY_RANGE=2310 # 1*2*3*5*7*11
RETRY = 5

EXCLUDE_LABEL = [
    'address1',
    'address2',
    'phone',
    'industry',
    'industryDisp',
    'sector',
    'sectorDisp',
    'longBusinessSummary',
    'companyOfficers',
    'governanceEpochDate',
    'compensationAsOfEpochDate',
    'maxAge',
    'previousClose',
    'open',
    'dayLow',
    'dayHigh',
    'volume',
    'lastFiscalYearEnd',
    'nextFiscalYearEnd',
    'mostRecentQuarter',
    'underlyingSymbol',
    'longName',
    'timeZoneFullName',
    'timeZoneShortName',
    'gmtOffSetMilliseconds',
    'financialCurrency',
    'fax',
]

def usage():
    msg = """
    Usage : check_stock.py [OPTION]
    """
    print(msg)
    exit(0)


def show_version():
    print(VERSION)
    exit(0)


def debug(level, msg):
    if ( level <= VERBOSE ):
        print(f"[D{level}] {msg}")


def load_tickers(fname):
    with open(fname, 'r') as f:
        tickers = [line.rstrip() for line in f]        
    return tickers


def parse_options(argv):
    if ( len(argv) == 0 ):
        usage()

    config = { 
            'resume': 0,
            'ticker_head': None,
    }
    while( len(argv) > 0 ):
        opt = argv.pop(0)
        if ( opt == '-h' ):
            usage()
        elif ( opt == '-v' ):
            show_version()      
        elif ( opt == '-r' ):
            config['resume'] = int( argv.pop(0) )
        elif ( opt == '-w' ):
            config['ticker_head'] = argv.pop(0)

    return config
    

def timestamp2unix(date):
    unix = (date.astype(int) / 10**9).to_numpy()
    return unix


def get_x_y(tk, day_range=DAY_RANGE):
    # get history from df
    date_end = datetime.datetime.today()
    date_start = date_end - datetime.timedelta(days=DAY_RANGE)
    df = tk.history(start=date_start, end=date_end)

    # x: datetime, y: closing   price
    x = timestamp2unix( df.index )
    n = x.shape[0]
    if ( n < day_range ):
        if ( n > day_range/2 ):
            day_range /= 2
        elif ( n > day_range/3 ):
            day_range /= 3
        elif ( n > day_range/5 ):
            day_range /= 5
        elif ( n > day_range/7 ):
            day_range /= 7
        elif ( n > day_range/11 ):
            day_range /= 11
        elif ( n > day_range/77 ):
            day_range /= 77
        else:
            return None, None

        date_start = date_end - datetime.timedelta(days=day_range)
        df = tk.history(start=date_start, end=date_end)
        x = timestamp2unix( df.index )

    y = df['Close'].to_numpy()
    return x, y


def extend_x_y(x, y):
    x0 = x[0]
    dx = x[1] - x[0]
    xN = x[-1] - x0
    x__ = x - x0
    __x = x__[::-1]
    ext_x = np.concatenate((x__, -__x + 2*xN + dx, x__ + 2*xN + 2*dx, -__x + 4*xN + 3*dx ))
    
    y0 = y[0]
    y__ = y - y0
    __y = y__[::-1]
    ext_y = np.concatenate((y__, __y, -y__, -__y))
    return ext_x, ext_y


def scaler_x_y(x, y, T=1):
    x_min = x[0]
    x_max = x[-1]
    norm_x = ( x - x_min ) / (x_max - x_min ) * T

    y_mean = np.mean(y)
    y_std = np.std(y)
    norm_y = ( y - y_mean ) / y_std
    return x_min, x_max, norm_x, y_mean, y_std, norm_y


def fourier_encode(n, x, y, T=1):
    # T=2pi
    b = np.zeros(n)
    for k in range(n):
        y_star = y * np.sin(2*(k+1)*np.pi*x / T)
        b[k] = 2/T * integrate.trapz(y=y_star, x=x, axis=-1)
    return b
    

def fourier_decode_y(b, x, T=1):
    n = b.shape[0]
    y = np.zeros_like(x)
    for k in range(n):
        y += b[k] * np.sin(2*(k+1)*np.pi*x/T)
    return y


def fourier_decode_dy(b, x, T=1):
    n = b.shape[0]
    dy = np.zeros_like(x)
    for k in range(n):
        dy += b[k] * (2*(k+1)*np.pi/T) * np.cos(2*(k+1)*np.pi*x/T)
    return dy


def fourier_decode_ddy(b, x, T=1):
    n = b.shape[0]
    ddy = np.zeros_like(x)
    for k in range(n):
        ddy += b[k] * (2*(k+1)*np.pi/T)**2 * (-np.sin(2*(k+1)*np.pi*x/T))
    return ddy


def get_stock_info_by_ticker(ticker, n=55, T=1):
    info = {}
    tk = yf.Ticker( ticker )
    # about
    # info['country'] = tk.info['country']
    is_delisted = True
    for key, value in tk.info.items():
        if ( not key in EXCLUDE_LABEL ):
            info[ key ] = value
        if ( value is not None ):
            is_delisted *= False

    if ( is_delisted ):
        return None

    x, y = get_x_y(tk)
    if ( ( not x is None ) and ( x.shape[0] > 100 ) ):
        ext_x, ext_y = extend_x_y(x, y)
        x_min, x_max, norm_x, y_mean, y_std, norm_y = scaler_x_y(ext_x, ext_y, T)
        b = fourier_encode(n, norm_x, norm_y, T)

        d_max = x.shape[0]
        norm_decode_x = np.linspace(0, T/4, d_max)
        norm_decode_y = fourier_decode_y(b, norm_decode_x)
        norm_decode_dy = fourier_decode_dy(b, norm_decode_x)
        norm_decode_ddy = fourier_decode_ddy(b, norm_decode_x)
    
        decode_x = norm_decode_x / T * (x_max - x_min ) + x_min + x[0]
        decode_y = norm_decode_y * y_std + y_mean + y[0]
        decode_dy = norm_decode_dy * y_std
        decode_ddy = norm_decode_ddy * y_std

        """
        # debug
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(411)
        ax.plot(x, y)
        ax = fig.add_subplot(412)
        ax.plot(decode_x, decode_y)
        ax = fig.add_subplot(413)
        ax.plot(decode_x, decode_dy)
        ax = fig.add_subplot(414)
        ax.plot(decode_x, decode_ddy)
        plt.show()
        # """

        info['current_price'] = y[-1]
        info['current_gradient'] = decode_dy[-1]
        info['current_phase'] = decode_ddy[-1]

        maximum_k = np.argmax(np.abs(b)) + 1
        info['principle_cycle'] = d_max / maximum_k

        cumsum = decode_y[-1] - decode_y[-2]
        flag = 0
        i = 0
        dim = decode_y.shape[0]
        if ( cumsum > 0 ): # increase
            while ( ( flag == 0 ) and ( i < dim ) ):
                i += 1
                diff = decode_y[-i-1] - decode_y[-i-2]
                if ( diff > 0 ): # increase
                    cumsum += diff
                else:
                    flag = 1
        else:
            while ( ( flag == 0 ) and ( i < dim ) ):
                i += 1
                diff = decode_y[-i-1] - decode_y[-i-2]
                if ( diff < 0 ): # increase
                    cumsum += diff
                else:
                    flag = 1
        info['days since last reversal'] = i
        info['fluctuation since last reversal'] = cumsum

        # corr coef
        monotonically_increasing = np.linspace(np.min(decode_y), np.max(decode_y), d_max)
        r = np.corrcoef(decode_y, monotonically_increasing)[0, 1]
        info['correlation coef'] = r
    
    return info

if __name__ == '__main__':
    # debug
    test_ticker = None#'00677U.TW' 
    if ( test_ticker is not None ):
        info = get_stock_info_by_ticker(test_ticker)
        print(info)
        exit(1)
    #
    tmpfile = '.list_dict_in_progress'
    csvfile = 'stock_check.csv'
    pickle_file  = 'stock_check.pickle'
    argv = sys.argv[1:]
    config = parse_options(argv)
    
    if ( config['resume'] <= 0 ):
        list_dict = {}
    else:
        with open(tmpfile, 'rb') as f:
           list_dict = pickle.load( f )

    config_file = CONFIG_FILE
    if ( config['ticker_head'] is not None ):
        word = config['ticker_head']
        tmpfile = tmpfile + f'.{word}'
        csvfile = f'stock_check_{word}.csv'
        pickle_file  = f'stock_check_{word}.pickle'
        config_file = config_file + f'.{word}'

    df_key_all = {}
    tickers = load_tickers( config_file )

    listed_symbols = 0
    for i, ticker in enumerate(tickers):
        debug(0, (i, ticker))
        if ( config['resume'] >= i):
            continue

        for _ in range(RETRY):
            try:
                info = get_stock_info_by_ticker(ticker)
            except Exception as e:
                print('Retry occured')
                print(traceback.format_exc())
                time.sleep(1)
            else:
                break
            
        else:
            print('Retry count over')
            continue

        if ( info is None ):
            continue
        else:
            listed_symbols += 1

        for key, value in info.items():
            if ( not key in df_key_all ):
                df_key_all[ key ] = 1
                list_dict[ key ] = ['' for _ in range(listed_symbols-1)]
            list_dict[ key ].append(value)
        
        for key in df_key_all.keys():
            if ( not key in info ):
                list_dict[ key ].append('')

        if ( i % 1000 == 0 ):
            num_tmp = f'{tmpfile}.{i}'
            with open(num_tmp, 'wb') as f:
                print(f'You can resume process from here by "-r {i+1}"')
                pickle.dump(list_dict, f)

    df = pd.DataFrame(
            data=list_dict,
            columns=list_dict.keys(),
    )

    print(df)
    df.to_csv(csvfile)
    df.to_pickle(pickle_file)





