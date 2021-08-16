import numpy as np
import pandas as pd

"""
Usage : 
    data = {
        "data": {
            "candles": [
                ["05-09-2013", 5553.75, 5625.75, 5552.700195, 5592.950195, 274900],
                ["06-09-2013", 5617.450195, 5688.600098, 5566.149902, 5680.399902, 253000],
                ["10-09-2013", 5738.5, 5904.850098, 5738.200195, 5896.75, 275200],
                ["11-09-2013", 5887.25, 5924.350098, 5832.700195, 5913.149902, 265000],
                ["12-09-2013", 5931.149902, 5932, 5815.799805, 5850.700195, 273000],
                ...
                ["27-01-2014", 6186.299805, 6188.549805, 6130.25, 6135.850098, 190400],
                ["28-01-2014", 6131.850098, 6163.600098, 6085.950195, 6126.25, 184100],
                ["29-01-2014", 6161, 6170.450195, 6109.799805, 6120.25, 146700],
                ["30-01-2014", 6067, 6082.850098, 6027.25, 6073.700195, 208100],
                ["31-01-2014", 6082.75, 6097.850098, 6067.350098, 6089.5, 146700]        
            ]
        }
    }
    
    # Date must be present as a Pandas DataFrame with ['date', 'open', 'high', 'low', 'close', 'volume'] as columns
    df = pd.DataFrame(data["data"]["candles"], columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    
    # Columns as added by each function specific to their computations
    EMA(df, 'close', 'ema_5', 5)
    ATR(df, 14)
    SuperTrend(df, 10, 3)
    MACD(df)
"""

def first_letter_upper(df):
    df_temp=df.copy()
    
    for col in df_temp.columns:
        col_before=col
        col_after=str(col).upper()[0:1]+str(col)[1:len(col)]
        df_temp.rename(columns={col_before:col_after},inplace=True)
    
    return df_temp

def first_letter_upper_v2(df):
    df_temp=df.copy()
    
    for col in df_temp.columns:
        col_before=col
        col_after=str(col).upper()[0:1]+str(col).lower()[1:len(col)]
        df_temp.rename(columns={col_before:col_after},inplace=True)
    
    return df_temp

def first_letter_lower(df):
    df_temp=df.copy()
    
    for col in df_temp.columns:
        col_before=col
        col_after=str(col).lower()[0:1]+str(col).lower()[1:len(col)]
        df_temp.rename(columns={col_before:col_after},inplace=True)
    
    return df_temp

def HA(df, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Heiken Ashi Candles (HA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Heiken Ashi Close (HA_$ohlc[3])
            Heiken Ashi Open (HA_$ohlc[0])
            Heiken Ashi High (HA_$ohlc[1])
            Heiken Ashi Low (HA_$ohlc[2])
    """

    ha_open = 'HA_' + ohlc[0]
    ha_high = 'HA_' + ohlc[1]
    ha_low = 'HA_' + ohlc[2]
    ha_close = 'HA_' + ohlc[3]
    
    df[ha_close] = (df[ohlc[0]] + df[ohlc[1]] + df[ohlc[2]] + df[ohlc[3]]) / 4

    df[ha_open] = 0.00
    for i in range(0, len(df)):
        if i == 0:
            df[ha_open].iat[i] = (df[ohlc[0]].iat[i] + df[ohlc[3]].iat[i]) / 2
        else:
            df[ha_open].iat[i] = (df[ha_open].iat[i - 1] + df[ha_close].iat[i - 1]) / 2
            
    df[ha_high]=df[[ha_open, ha_close, ohlc[1]]].max(axis=1)
    df[ha_low]=df[[ha_open, ha_close, ohlc[2]]].min(axis=1)

    return df

def SMA(df, base, target, period):
    """
    Function to compute Simple Moving Average (SMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the SMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    df[target] = df[base].rolling(window=period).mean()
    df[target].fillna(0, inplace=True)

    return df

def rainbow_sma(df,base):
    short_lookback=2
    long_lookback=10
    
    ma1=df[base].rolling(window=short_lookback).mean()
    ma2=ma1.rolling(window=short_lookback).mean()
    ma3=ma2.rolling(window=short_lookback).mean()
    ma4=ma3.rolling(window=short_lookback).mean()
    ma5=ma4.rolling(window=short_lookback).mean()
    ma6=ma5.rolling(window=short_lookback).mean()
    ma7=ma6.rolling(window=short_lookback).mean()
    ma8=ma7.rolling(window=short_lookback).mean()
    ma9=ma8.rolling(window=short_lookback).mean()
    ma10=ma9.rolling(window=short_lookback).mean()
    hhh=df[base].rolling(window=long_lookback).max()
    lll=df[base].rolling(window=long_lookback).min()
    
    hhma=np.maximum(ma1,np.maximum(ma2,
                    np.maximum(ma3,np.maximum(ma4,np.maximum(ma5,
                    np.maximum(ma6,np.maximum(ma7,np.maximum(ma8,np.maximum(ma9,ma10)))))))))
    
    llma=np.minimum(ma1,np.minimum(ma2,
                    np.minimum(ma3,np.minimum(ma4,np.minimum(ma5,
                    np.minimum(ma6,np.minimum(ma7,np.minimum(ma8,np.minimum(ma9,ma10)))))))))
    
    ma_avg=(ma1+ma2+ma3+ma4+ma5+ma6+ma7+ma8+ma9+ma10)/10
    rbo=(df[base]-ma_avg)*100/(hhh-lll)
    rb=(hhma-llma)*100/(hhh-lll)
    
    df['hhma']=hhma
    df['llma']=llma
    
    
    df['ma_avg']=ma_avg
    df['rbo']=rbo
    df['rb']=rb
    
    return df

    
def STDDEV(df, base, target, period):
    """
    Function to compute Standard Deviation (STDDEV)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the SMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    df[target] = df[base].rolling(window=period).std()
    df[target].fillna(0, inplace=True)

    return df

def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])
    
    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()
    
    df[target].fillna(0, inplace=True)
    return df

def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Average True Range (ATR)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR_' + str(period)

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())
         
        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
         
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=True)
    
    return df

def SuperTrend(df, period, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute SuperTrend
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the ATR
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR), ATR (ATR_$period)
            SuperTrend (ST_$period_$multiplier)
            SuperTrend Direction (STX_$period_$multiplier)
    """
    df=first_letter_upper(df)
    ATR(df, period, ohlc=ohlc)
    atr = 'ATR_' + str(period)
    st = 'ST_' + str(period) + '_' + str(multiplier)
    stx = 'STX_' + str(period) + '_' + str(multiplier)
    
    """
    SuperTrend Algorithm :
    
        BASIC UPPERBAND = (HIGH + LOW) / 2 + Multiplier * ATR
        BASIC LOWERBAND = (HIGH + LOW) / 2 - Multiplier * ATR
        
        FINAL UPPERBAND = IF( (Current BASICUPPERBAND < Previous FINAL UPPERBAND) or (Previous Close > Previous FINAL UPPERBAND))
                            THEN (Current BASIC UPPERBAND) ELSE Previous FINALUPPERBAND)
        FINAL LOWERBAND = IF( (Current BASIC LOWERBAND > Previous FINAL LOWERBAND) or (Previous Close < Previous FINAL LOWERBAND)) 
                            THEN (Current BASIC LOWERBAND) ELSE Previous FINAL LOWERBAND)
        
        SUPERTREND = IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close <= Current FINAL UPPERBAND)) THEN
                        Current FINAL UPPERBAND
                    ELSE
                        IF((Previous SUPERTREND = Previous FINAL UPPERBAND) and (Current Close > Current FINAL UPPERBAND)) THEN
                            Current FINAL LOWERBAND
                        ELSE
                            IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close >= Current FINAL LOWERBAND)) THEN
                                Current FINAL LOWERBAND
                            ELSE
                                IF((Previous SUPERTREND = Previous FINAL LOWERBAND) and (Current Close < Current FINAL LOWERBAND)) THEN
                                    Current FINAL UPPERBAND
    """
    
    # Compute basic upper and lower bands
    df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
    df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]

    # Compute final upper and lower bands
    df['final_ub'] = 0.00
    df['final_lb'] = 0.00
    for i in range(period, len(df)):
        df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['Close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
        df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['Close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]
       
    # Set the Supertrend value
    df[st] = 0.00
    for i in range(period, len(df)):
        df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['Close'].iat[i] <= df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['Close'].iat[i] >  df['final_ub'].iat[i] else \
                        df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[i] >= df['final_lb'].iat[i] else \
                        df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['Close'].iat[i] <  df['final_lb'].iat[i] else 0.00 
                 
    # Mark the trend direction up/down
    df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down',  'up'), np.NaN)

    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
    
    df.fillna(0, inplace=True)

    return df

def MACD(df, fastEMA=12, slowEMA=26, signal=9, base='Close'):
    """
    Function to compute Moving Average Convergence Divergence (MACD)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        fastEMA : Integer indicates faster EMA
        slowEMA : Integer indicates slower EMA
        signal : Integer indicates the signal generator for MACD
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Fast EMA (ema_$fastEMA)
            Slow EMA (ema_$slowEMA)
            MACD (macd_$fastEMA_$slowEMA_$signal)
            MACD Signal (signal_$fastEMA_$slowEMA_$signal)
            MACD Histogram (MACD (hist_$fastEMA_$slowEMA_$signal)) 
    """

    fE = "ema_" + str(fastEMA)
    sE = "ema_" + str(slowEMA)
    macd = "macd_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    sig = "signal_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)
    hist = "hist_" + str(fastEMA) + "_" + str(slowEMA) + "_" + str(signal)

    # Compute fast and slow EMA    
    EMA(df, base, fE, fastEMA)
    EMA(df, base, sE, slowEMA)
    
    # Compute MACD
    df[macd] = np.where(np.logical_and(np.logical_not(df[fE] == 0), np.logical_not(df[sE] == 0)), df[fE] - df[sE], 0)
    
    # Compute MACD Signal
    EMA(df, macd, sig, signal)
    
    # Compute MACD Histogram
    df[hist] = np.where(np.logical_and(np.logical_not(df[macd] == 0), np.logical_not(df[sig] == 0)), df[macd] - df[sig], 0)
    
    return df

def BBand(df, base='Close', period=20, multiplier=2):
    """
    Function to compute Bollinger Band (BBand)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        period : Integer indicates the period of computation in terms of number of candles
        multiplier : Integer indicates value to multiply the SD
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Upper Band (UpperBB_$period_$multiplier)
            Lower Band (LowerBB_$period_$multiplier)
    """
    
    upper = 'UpperBB_' + str(period) + '_' + str(multiplier)
    lower = 'LowerBB_' + str(period) + '_' + str(multiplier)
    
    sma = df[base].rolling(window=period, min_periods=period - 1).mean()
    sd = df[base].rolling(window=period).std()
    df[upper] = sma + (multiplier * sd)
    df[lower] = sma - (multiplier * sd)
    
    df[upper].fillna(0, inplace=True)
    df[lower].fillna(0, inplace=True)
    
    return df

def RSI(df, base="Close", period=21):
    """
    Function to compute Relative Strength Index (RSI)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Relative Strength Index (RSI_$period)
    """
    #df=first_letter_upper(df)
    delta = df[base].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    rUp = up.ewm(com=period - 1,  adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()

    df['RSI_' + str(period)] = 100 - 100 / (1 + rUp / rDown)
    df['RSI_' + str(period)].fillna(0, inplace=True)

    return df

def Ichimoku(df, ohlc=['Open', 'High', 'Low', 'Close'], param=[9, 26, 52, 26]):
    """
    Function to compute Ichimoku Cloud parameter (Ichimoku)
    
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        param: Periods to be used in computation (default [tenkan_sen_period, kijun_sen_period, senkou_span_period, chikou_span_period] = [9, 26, 52, 26])
        
    Returns :
        df : Pandas DataFrame with new columns added for ['Tenkan Sen', 'Kijun Sen', 'Senkou Span A', 'Senkou Span B', 'Chikou Span']
    """
    
    high = df[ohlc[1]]
    low = df[ohlc[2]]
    close = df[ohlc[3]]
    
    tenkan_sen_period = param[0]
    kijun_sen_period = param[1]
    senkou_span_period = param[2]
    chikou_span_period = param[3]
    
    tenkan_sen_column = 'Tenkan Sen'
    kijun_sen_column = 'Kijun Sen'
    senkou_span_a_column = 'Senkou Span A'
    senkou_span_b_column = 'Senkou Span B'
    chikou_span_column = 'Chikou Span'
    
    # Tenkan-sen (Conversion Line)
    tenkan_sen_high = high.rolling(window=tenkan_sen_period).max()
    tenkan_sen_low = low.rolling(window=tenkan_sen_period).min()
    df[tenkan_sen_column] = (tenkan_sen_high + tenkan_sen_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_sen_high = high.rolling(window=kijun_sen_period).max()
    kijun_sen_low = low.rolling(window=kijun_sen_period).min()
    df[kijun_sen_column] = (kijun_sen_high + kijun_sen_low) / 2
    
    # Senkou Span A (Leading Span A)
    df[senkou_span_a_column] = ((df[tenkan_sen_column] + df[kijun_sen_column]) / 2).shift(kijun_sen_period)
    
    # Senkou Span B (Leading Span B)
    senkou_span_high = high.rolling(window=senkou_span_period).max()
    senkou_span_low = low.rolling(window=senkou_span_period).min()
    df[senkou_span_b_column] = ((senkou_span_high + senkou_span_low) / 2).shift(kijun_sen_period)
    
    # The most current closing price plotted chikou_span_period time periods behind
    df[chikou_span_column] = close.shift(-1 * chikou_span_period)
    
    return df

#def candle_score(lst_0=[O_0,H_0,L_0,C_0],lst_1=[O_1,H_1,L_1,C_1],lst_2=[O_2,H_2,L_2,C_2]):    
def candle_score(lst_0,lst_1,lst_2):    
    
    O_0,H_0,L_0,C_0=lst_0[0],lst_0[1],lst_0[2],lst_0[3]
    O_1,H_1,L_1,C_1=lst_1[0],lst_1[1],lst_1[2],lst_1[3]
    O_2,H_2,L_2,C_2=lst_2[0],lst_2[1],lst_2[2],lst_2[3]
    
    DojiSize = 0.1
    
    doji=(abs(O_0 - C_0) <= (H_0 - L_0) * DojiSize)
    
    strength=((C_0 - O_0) <= ((H_0 - L_0) * 0.5))
    
    hammer=(((H_0 - L_0)>3*abs(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))
    
    hammer_pin=(((H_0 - L_0)>15*(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))

    
    inverted_hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((H_0 - C_0)/(.001 + H_0 - L_0) > 0.6) & ((H_0 - O_0)/(.001 + H_0 - L_0) > 0.6))
    
    bullish_reversal= (O_2 > C_2)&(O_1 > C_1)&doji
    
    bearish_reversal= (O_2 < C_2)&(O_1 < C_1)&doji
    
    evening_star=(C_2 > O_2) & (min(O_1, C_1) > C_2) & (O_0 < min(O_1, C_1)) & (C_0 < O_0 )
    
    morning_star=(C_2 < O_2) & (min(O_1, C_1) < C_2) & (O_0 > min(O_1, C_1)) & (C_0 > O_0 )
    
    shooting_Star_bearish=(O_1 < C_1) & (O_0 > C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    shooting_Star_bullish=(O_1 > C_1) & (O_0 < C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    bearish_harami=(C_1 > O_1) & (O_0 > C_0) & (O_0 <= C_1) & (O_1 <= C_0) & ((O_0 - C_0) < (C_1 - O_1 ))
    
    Bullish_Harami=(O_1 > C_1) & (C_0 > O_0) & (C_0 <= O_1) & (C_1 <= O_0) & ((C_0 - O_0) < (O_1 - C_1))
    
    Bearish_Engulfing=((C_1 > O_1) & (O_0 > C_0)) & ((O_0 >= C_1) & (O_1 >= C_0)) & ((O_0 - C_0) > (C_1 - O_1 ))
    
    Bullish_Engulfing=(O_1 > C_1) & (C_0 > O_0) & (C_0 >= O_1) & (C_1 >= O_0) & ((C_0 - O_0) > (O_1 - C_1 ))
    
    Piercing_Line_bullish=(C_1 < O_1) & (C_0 > O_0) & (O_0 < L_1) & (C_0 > C_1)& (C_0>((O_1 + C_1)/2)) & (C_0 < O_1)

    Hanging_Man_bullish=(C_1 < O_1) & (O_0 < L_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Hanging_Man_bearish=(C_1 > O_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Green=(C_0 > O_0)
    
    Red=(C_0 <= O_0)

    Gap_up_15=(((C_1-C_0)*100/C_0)>1.5)
    Gap_dn_15=(((C_1-C_0)*100/C_0)<-1.5)

    Green=(C_0 > O_0)
    Green_1=(C_1 > O_1)
    
    Red_0=(C_0 <= O_0)
    Red_1=(C_1 <= O_1)
    Red_2=(C_2 <= O_2)

    Gap_up_15_0=(((C_0-C_1)*100/C_0)>1.5)&(((O_0-O_1)*100/O_0)>1.5)

    Gap_dn_15_0=(((C_0-C_1)*100/C_0)<-1.5)&(((O_0-O_1)*100/O_0)<-1.5)


    Gap_up_15_1=(((C_1-C_2)*100/C_1)>1.5)&(((O_1-O_2)*100/O_1)>1.5)

    Gap_dn_15_1=(((C_1-C_2)*100/C_1)<-1.5)&(((O_1-O_2)*100/O_1)<-1.5)
    
    
    
    MarubozuSize=0.50
    Marubozu_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize)
    Marubozu_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize)
    Marubozu_2=(abs(O_2 - C_2) >= (H_2 - L_2) * MarubozuSize)
    
    Gap_up_95=(((C_0-C_1)*100/C_0)>0.5)&(((O_0-O_1)*100/O_0)>0.5)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<-0.5)&(((O_0-O_1)*100/O_0)<-0.5)

    Red_1=(C_1 < O_1)
    Kicking_minus=Red&Green_1&Marubozu_0&Marubozu_1
    
    kick_minus_satyarth_cond_1=(O_1 > O_0)
    
    Kicking_minus=Green_1&kick_minus_satyarth_cond_1

    Three_black_crows=Red_0&Red_1&Red_2
    Three_Marubozu=Marubozu_0&Marubozu_1&Marubozu_2
    body_ratio0=abs(O_0 - C_0)/(H_0 - L_0)
    body_ratio1=abs(O_1 - C_1)/(H_1 - L_1)
    body_ratio2=abs(O_1 - C_2)/(H_2 - L_2)

    Exciting0=(body_ratio0>= 0.5)
    
    Exciting1=(body_ratio1>= 0.5)
    Exciting2=(body_ratio2>= 0.5)
    
    Exciting=Exciting0|Exciting1|Exciting2
    Exciting=Exciting0

    Boring0=(body_ratio0<0.5)
    Boring1=(body_ratio1<0.5)
    Boring2=(body_ratio2<0.5)
    
    Boring=Boring0|Boring1|Boring2
    Boring=Boring0
    
    strCandle=''
    candle_score=0


    if Green:
        strCandle=strCandle+'/ '+'Green'
        candle_score=candle_score+0

    if Red:
        strCandle=strCandle+'/ '+'Red'

    if Gap_up_15_0:
        strCandle=strCandle+'/ '+'Gap_up_15'
        candle_score=candle_score+0

    if Gap_dn_15_0:
        strCandle=strCandle+'/ '+'Gap_dn_15'
        candle_score=candle_score-0
        
    if doji:
        strCandle=strCandle+'/ '+'doji'
        candle_score=candle_score+0

    if Exciting:
        strCandle=strCandle+'/ '+'Exciting'
        candle_score=candle_score+0

    if Boring:
        strCandle=strCandle+'/ '+'Boring'
        candle_score=candle_score-0

    if strength:
        strCandle=strCandle+'/ '+'strength'

    if hammer_pin:
        strCandle=strCandle+'/ '+'hammer_pin'
        candle_score=candle_score-0

        
    '''
    if morning_star:
        strCandle=strCandle+'/ '+'morning_star'
        candle_score=candle_score+1
    if shooting_Star_bullish:
        strCandle=strCandle+'/ '+'shooting_Star_bullish'
        candle_score=candle_score+1
    if    Bullish_Harami:
        strCandle=strCandle+'/ '+'Bullish_Harami'
        candle_score=candle_score+1
    if    Bullish_Engulfing:
        strCandle=strCandle+'/ '+'Bullish_Engulfing'
        candle_score=candle_score+1
    if    bullish_reversal:
        strCandle=strCandle+'/ '+'bullish_reversal'
        candle_score=candle_score+1
    if    Piercing_Line_bullish:
        strCandle=strCandle+'/ '+'Piercing_Line_bullish'
        candle_score=candle_score+1
    if    Hanging_Man_bullish:
        strCandle=strCandle+'/ '+'Hanging_Man_bullish'
        candle_score=candle_score+1
    '''

    if Kicking_minus:
        strCandle=strCandle+'/ '+'Kicking_minus'
        candle_score=candle_score-1

    if evening_star:
        strCandle=strCandle+'/ '+'evening_star'
        candle_score=candle_score-1

    if shooting_Star_bearish:
        strCandle=strCandle+'/ '+'shooting_Star_bearish'
        candle_score=candle_score
    if    hammer:
        strCandle=strCandle+'/ '+'hammer'
    if    inverted_hammer:
        strCandle=strCandle+'/ '+'inverted_hammer'
    if    bearish_harami:
        strCandle=strCandle+'/ '+'bearish_harami'
        candle_score=candle_score-1
    if    Bearish_Engulfing:
        strCandle=strCandle+'/ '+'Bearish_Engulfing'
        candle_score=candle_score-1
    if    bearish_reversal:
        strCandle=strCandle+'/ '+'bearish_reversal'
        candle_score=candle_score-1
    if    Hanging_Man_bearish:
        strCandle=strCandle+'/ '+'Hanging_Man_bearish'
        candle_score=candle_score-1
    '''
    if Three_black_crows:
        strCandle=strCandle+'/ '+'Three_black_crows'
        candle_score=candle_score-1
        
    if Three_Marubozu:
        strCandle=strCandle+'/ '+'Three_Marubozu'
        candle_score=candle_score-1
    '''                
        
    #return candle_score
    return candle_score,strCandle
def candle_score2(lst_0,lst_1,lst_2,type):    
    
    O_0,H_0,L_0,C_0=lst_0[0],lst_0[1],lst_0[2],lst_0[3]
    O_1,H_1,L_1,C_1=lst_1[0],lst_1[1],lst_1[2],lst_1[3]
    O_2,H_2,L_2,C_2=lst_2[0],lst_2[1],lst_2[2],lst_2[3]
    
    DojiSize = 0.1
    
    doji=(abs(O_0 - C_0) <= (H_0 - L_0) * DojiSize)
    
    doji_1=(abs(O_1 - C_1) <= (H_1 - L_1) * DojiSize)
    
    
    strength=((C_0 - O_0) <= ((H_0 - L_0) * 0.5))
    
    hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))
    
    inverted_hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((H_0 - C_0)/(.001 + H_0 - L_0) > 0.6) & ((H_0 - O_0)/(.001 + H_0 - L_0) > 0.6))
    
    bullish_reversal= (O_2 > C_2)&(O_1 > C_1)&doji
    
    bearish_reversal= (O_2 < C_2)&(O_1 < C_1)&doji
    
    evening_star=(C_2 > O_2) & (min(O_1, C_1) > C_2) & (O_0 < min(O_1, C_1)) & (C_0 < O_0 )
    
    morning_star=(C_2 < O_2) & (min(O_1, C_1) < C_2) & (O_0 > min(O_1, C_1)) & (C_0 > O_0 )
    
    shooting_Star_bearish=(O_1 < C_1) & (O_0 > C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    shooting_Star_bullish=(O_1 > C_1) & (O_0 < C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    bearish_harami=(C_1 > O_1) & (O_0 > C_0) & (O_0 <= C_1) & (O_1 <= C_0) & ((O_0 - C_0) < (C_1 - O_1 ))
    
    Bullish_Harami=(O_1 > C_1) & (C_0 > O_0) & (C_0 <= O_1) & (C_1 <= O_0) & ((C_0 - O_0) < (O_1 - C_1))
    
    Bearish_Engulfing=((C_1 > O_1) & (O_0 > C_0)) & ((O_0 >= C_1) & (O_1 >= C_0)) & ((O_0 - C_0) > (C_1 - O_1 ))
    
    Bullish_Engulfing=(O_1 > C_1) & (C_0 > O_0) & (C_0 >= O_1) & (C_1 >= O_0) & ((C_0 - O_0) > (O_1 - C_1 ))
    
    Piercing_Line_bullish=(C_1 < O_1) & (C_0 > O_0) & (O_0 < L_1) & (C_0 > C_1)& (C_0>((O_1 + C_1)/2)) & (C_0 < O_1)

    Hanging_Man_bullish=(C_1 < O_1) & (O_0 < L_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Hanging_Man_bearish=(C_1 > O_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Green=(C_0 > O_0)
    Green_0=(C_0 > O_0)
    Green_1=(C_1 > O_1)
    Green_2=(C_2 > O_2)
    
    Red=(C_0 <= O_0)

    Gap_up_15_0=(((C_0-C_1)*100/C_0)>1.5)&(((O_0-O_1)*100/O_0)>1.5)

    Gap_dn_15_0=(((C_0-C_1)*100/C_0)<-1.5)&(((O_0-O_1)*100/O_0)<-1.5)


    Gap_up_15_1=(((C_1-C_2)*100/C_1)>1.5)&(((O_1-O_2)*100/O_1)>1.5)

    Gap_dn_15_1=(((C_1-C_2)*100/C_1)<-1.5)&(((O_1-O_2)*100/O_1)<-1.5)
    
    
    
    Abandoned_baby=doji_1&Gap_dn_15_1&Gap_up_15_0&Green
    MarubozuSize=0.50
    Marubozu_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize)
    Marubozu_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize)
    Marubozu_2=(abs(O_2 - C_2) >= (H_2 - L_2) * MarubozuSize)

    gap_up_size=0.35
    gap_dn_size=-0.25

    Gap_up_95=(((C_0-C_1)*100/C_0)>gap_up_size)&(((O_0-O_1)*100/O_0)>gap_up_size)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<gap_dn_size)&(((O_0-O_1)*100/O_0)<gap_dn_size)

    Red_1=(C_1 < O_1)
    kick_satyarth_cond_1=(O_0 > O_1)

    Kicking_plus_van=Green&Red_1&Gap_up_95&Marubozu_0&Marubozu_1
    Kicking_plus=Gap_up_95&Marubozu_0&Marubozu_1
    Kicking_plus_satyarth=Red_1&kick_satyarth_cond_1

    Kicking_minus=Red&Green_1&Gap_dn_95&Marubozu_0&Marubozu_1

    gap_up_size_pg=0.25
    Gap_up_pg=(((C_0-C_1)*100/C_0)>gap_up_size_pg)&(((O_0-O_1)*100/O_0)>gap_up_size_pg)
    
    body_ratio0=abs(O_0 - C_0)/(H_0 - L_0)
    body_ratio1=abs(O_1 - C_1)/(H_1 - L_1)
    body_ratio2=abs(O_1 - C_2)/(H_2 - L_2)

    Exciting0=(body_ratio0>= 0.5)
    
    Exciting1=(body_ratio1>= 0.5)
    Exciting2=(body_ratio2>= 0.5)
    
    Exciting=Exciting0|Exciting1|Exciting2
    Exciting=Exciting0

    Boring0=(body_ratio0<0.5)
    Boring1=(body_ratio1<0.5)
    Boring2=(body_ratio2<0.5)
    
    Boring=Boring0|Boring1|Boring2
    Boring=Boring0
    
    strCandle=''
    candle_score=0

    
    
    Red_1=(C_1 < O_1)
    Kicking_minus=Red&Green_1&Gap_dn_95&Marubozu_0&Marubozu_1
    
    Three_white_crows=Green_0&Green_1&Green_2
    Three_Marubozu=Marubozu_0&Marubozu_1&Marubozu_2
    
    Kicking_plus_sunil=Green_0&Red_1&kick_satyarth_cond_1&Gap_up_15_0&(morning_star|(Three_Marubozu&Three_white_crows))


    if Green:
        strCandle=strCandle+'/ '+'Green'
        candle_score=candle_score+0

    if Red:
        strCandle=strCandle+'/ '+'Red'

    if Red_1:
        strCandle=strCandle+'/ '+'Red_1'

    if Gap_up_15_0:
        strCandle=strCandle+'/ '+'Gap_up_15'
        candle_score=candle_score+0

    if Gap_dn_15_0:
        strCandle=strCandle+'/ '+'Gap_dn_15'
        candle_score=candle_score-0
        
    if doji:
        strCandle=strCandle+'/ '+'doji'
        candle_score=candle_score+0

    if Exciting:
        strCandle=strCandle+'/ '+'Exciting'
        candle_score=candle_score+0

    if Boring:
        strCandle=strCandle+'/ '+'Boring'
        candle_score=candle_score-0

    if strength:
        strCandle=strCandle+'/ '+'strength'

    if Abandoned_baby:
        strCandle=strCandle+'/ '+'Abandoned_baby'
        
        
    if type=='bull':
        if Kicking_plus:
            strCandle=strCandle+'/ '+'Kicking_plus'
            candle_score=candle_score+100

        if Kicking_plus_satyarth:
            strCandle=strCandle+'/ '+'Kicking_plus_satyarth'
            candle_score=candle_score+100
            

        if Kicking_plus_sunil:
            strCandle=strCandle+'/ '+'Kicking_plus_sunil'
            candle_score=candle_score+100
            
        if Kicking_plus_van:
            strCandle=strCandle+'/ '+'Kicking_plus_van'
            candle_score=candle_score+10

        if morning_star:
            strCandle=strCandle+'/ '+'morning_star'
            candle_score=candle_score+1
        if shooting_Star_bullish:
            strCandle=strCandle+'/ '+'shooting_Star_bullish'
            candle_score=candle_score+1
        if    Bullish_Harami:
            strCandle=strCandle+'/ '+'Bullish_Harami'
            candle_score=candle_score+1
        if    Bullish_Engulfing:
            strCandle=strCandle+'/ '+'Bullish_Engulfing'
            candle_score=candle_score+1
        if    bullish_reversal:
            strCandle=strCandle+'/ '+'bullish_reversal'
            candle_score=candle_score+1
        if    Piercing_Line_bullish:
            strCandle=strCandle+'/ '+'Piercing_Line_bullish'
            candle_score=candle_score+1
        if    Hanging_Man_bullish:
            strCandle=strCandle+'/ '+'Hanging_Man_bullish'
            candle_score=candle_score+1
        if    Three_Marubozu:
            strCandle=strCandle+'/ '+'Three_Marubozu'
            candle_score=candle_score+1
        if    Three_white_crows:
            strCandle=strCandle+'/ '+'Three_white_crows'
            candle_score=candle_score+1

    if type!='bull':
        if Kicking_minus:
            strCandle=strCandle+'/ '+'Kicking_minus'
            candle_score=candle_score-1

        
        if evening_star:
            strCandle=strCandle+'/ '+'evening_star'
            candle_score=candle_score-1
    
        if shooting_Star_bearish:
            strCandle=strCandle+'/ '+'shooting_Star_bearish'
            candle_score=candle_score-1
        if    hammer:
            strCandle=strCandle+'/ '+'hammer'
        if    inverted_hammer:
            strCandle=strCandle+'/ '+'inverted_hammer'
        if    bearish_harami:
            strCandle=strCandle+'/ '+'bearish_harami'
            candle_score=candle_score-1
        if    Bearish_Engulfing:
            strCandle=strCandle+'/ '+'Bearish_Engulfing'
            candle_score=candle_score-1
        if    bearish_reversal:
            strCandle=strCandle+'/ '+'bearish_reversal'
            candle_score=candle_score-1
        if    Hanging_Man_bearish:
            strCandle=strCandle+'/ '+'Hanging_Man_bearish'
            candle_score=candle_score-1
            
            
    #return candle_score
    return candle_score,strCandle

def candle_score_v4(lst_0,lst_1,lst_2):    
    
    O_0,H_0,L_0,C_0=lst_0[0],lst_0[1],lst_0[2],lst_0[3]
    O_1,H_1,L_1,C_1=lst_1[0],lst_1[1],lst_1[2],lst_1[3]
    O_2,H_2,L_2,C_2=lst_2[0],lst_2[1],lst_2[2],lst_2[3]
    
    DojiSize = 0.1
    
    doji=(abs(O_0 - C_0) <= (H_0 - L_0) * DojiSize)
    
    doji_1=(abs(O_1 - C_1) <= (H_1 - L_1) * DojiSize)
    
    
    strength=((C_0 - O_0) <= ((H_0 - L_0) * 0.5))
    
    hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))
    
    inverted_hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((H_0 - C_0)/(.001 + H_0 - L_0) > 0.6) & ((H_0 - O_0)/(.001 + H_0 - L_0) > 0.6))
    
    bullish_reversal= (O_2 > C_2)&(O_1 > C_1)&doji
    
    bearish_reversal= (O_2 < C_2)&(O_1 < C_1)&doji
    
    evening_star=(C_2 > O_2) & (min(O_1, C_1) > C_2) & (O_0 < min(O_1, C_1)) & (C_0 < O_0 )
    
    morning_star=(C_2 < O_2) & (min(O_1, C_1) < C_2) & (O_0 > min(O_1, C_1)) & (C_0 > O_0 )
    
    shooting_Star_bearish=(O_1 < C_1) & (O_0 > C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    shooting_Star_bullish=(O_1 > C_1) & (O_0 < C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    bearish_harami=(C_1 > O_1) & (O_0 > C_0) & (O_0 <= C_1) & (O_1 <= C_0) & ((O_0 - C_0) < (C_1 - O_1 ))
    
    Bullish_Harami=(O_1 > C_1) & (C_0 > O_0) & (C_0 <= O_1) & (C_1 <= O_0) & ((C_0 - O_0) < (O_1 - C_1))
    
    Bearish_Engulfing=((C_1 > O_1) & (O_0 > C_0)) & ((O_0 >= C_1) & (O_1 >= C_0)) & ((O_0 - C_0) > (C_1 - O_1 ))
    
    Bullish_Engulfing=(O_1 > C_1) & (C_0 > O_0) & (C_0 >= O_1) & (C_1 >= O_0) & ((C_0 - O_0) > (O_1 - C_1 ))
    
    Piercing_Line_bullish=(C_1 < O_1) & (C_0 > O_0) & (O_0 < L_1) & (C_0 > C_1)& (C_0>((O_1 + C_1)/2)) & (C_0 < O_1)

    Hanging_Man_bullish=(C_1 < O_1) & (O_0 < L_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Hanging_Man_bearish=(C_1 > O_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Green=(C_0 > O_0)
    Green_0=(C_0 > O_0)
    Green_1=(C_1 > O_1)
    Green_2=(C_2 > O_2)
    
    Red=(C_0 <= O_0)
    
    Red_0=(C_0 <= O_0)
    Red_1=(C_1 <= O_1)
    Red_2=(C_2 <= O_2)

    Gap_up_15_0=(((C_0-C_1)*100/C_0)>1.5)&(((O_0-O_1)*100/O_0)>1.5)

    Gap_dn_15_0=(((C_0-C_1)*100/C_0)<-1.5)&(((O_0-O_1)*100/O_0)<-1.5)


    Gap_up_15_1=(((C_1-C_2)*100/C_1)>1.5)&(((O_1-O_2)*100/O_1)>1.5)

    Gap_dn_15_1=(((C_1-C_2)*100/C_1)<-1.5)&(((O_1-O_2)*100/O_1)<-1.5)
    
    
    
    Abandoned_baby=doji_1&Gap_dn_15_1&Gap_up_15_0&Green
    MarubozuSize=0.50
    Marubozu_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize)
    Marubozu_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize)
    
    MarubozuSize=0.50
    Marubozu_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize)
    Marubozu_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize)
    Marubozu_2=(abs(O_2 - C_2) >= (H_2 - L_2) * MarubozuSize)
    
    Gap_up_95=(((C_0-C_1)*100/C_0)>0.5)&(((O_0-O_1)*100/O_0)>0.5)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<-0.5)&(((O_0-O_1)*100/O_0)<-0.5)

    Red_1=(C_1 < O_1)
    Kicking_minus=Red&Green_1&Gap_dn_95&Marubozu_0&Marubozu_1
    Three_white_crows=Green_0&Green_1&Green_2

    Three_black_crows=Red_0&Red_1&Red_2
    Three_Marubozu=Marubozu_0&Marubozu_1&Marubozu_2

    Marubozu=Marubozu_0
    gap_up_size=0.35
    gap_dn_size=-0.25

    Gap_up_95=(((C_0-C_1)*100/C_0)>gap_up_size)&(((O_0-O_1)*100/O_0)>gap_up_size)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<gap_dn_size)&(((O_0-O_1)*100/O_0)<gap_dn_size)

    Red_1=(C_1 < O_1)
    Kicking_plus_van=Green&Red_1&Gap_up_95&Marubozu_0&Marubozu_1
    Kicking_plus=Gap_up_95&Marubozu_0&Marubozu_1
    
    kick_minus_satyarth_cond_1=(O_1 > O_0)
    
    Kicking_minus=Green_1&kick_minus_satyarth_cond_1


    gap_up_size_pg=0.25
    Gap_up_pg=(((C_0-C_1)*100/C_0)>gap_up_size_pg)&(((O_0-O_1)*100/O_0)>gap_up_size_pg)
    
    body_ratio0=abs(O_0 - C_0)/(H_0 - L_0)
    body_ratio1=abs(O_1 - C_1)/(H_1 - L_1)
    body_ratio2=abs(O_1 - C_2)/(H_2 - L_2)

    Exciting0=(body_ratio0>= 0.5)
    
    Exciting1=(body_ratio1>= 0.5)
    Exciting2=(body_ratio2>= 0.5)
    
    Exciting=Exciting0|Exciting1|Exciting2
    Exciting=Exciting0

    Boring0=(body_ratio0<0.5)
    Boring1=(body_ratio1<0.5)
    Boring2=(body_ratio2<0.5)
    
    Boring=Boring0|Boring1|Boring2
    Boring=Boring0
    
    strCandle=''
    candle_score=0


    lst_cols=['f_doji',
    'f_strength',
    'f_hammer',
    'f_inverted_hammer',
    'f_Green',
    'f_Gap_up_15_0',
    'f_Gap_dn_15_0',
    'f_Marubozu',
    'f_Gap_up_95',
    'f_Gap_dn_95',
    'f_Kicking_plus_van',
    'f_Kicking_plus',
    'f_Kicking_minus',
    'f_Gap_up_pg',
    'f_Exciting',
    'f_Boring',
    
    'f_morning_star',
    'f_shooting_Star_bullish',
    'f_Bullish_Harami',
    'f_Bullish_Engulfing',
    'f_bullish_reversal',
    'f_Piercing_Line_bullish',
    'f_Hanging_Man_bullish',
    'f_evening_star',
    'f_shooting_Star_bearish',
    'f_bearish_harami',
    'f_Bearish_Engulfing',
    'f_bearish_reversal',
    'f_Hanging_Man_bearish',
    'f_Three_white_crows',
    'f_Three_Marubozu',
    'f_Three_black_crows'
    
    ]


    lst_vals=[(1 if doji else 0),
    (1 if strength else 0),
    (1 if hammer else 0),
    (1 if inverted_hammer else 0),
    (1 if Green else 0),
    (1 if Gap_up_15_0 else 0),
    (1 if Gap_dn_15_0 else 0),
    (1 if Marubozu else 0),
    (1 if Gap_up_95 else 0),
    (1 if Gap_dn_95 else 0),
    (1 if Kicking_plus_van else 0),
    (1 if Kicking_plus else 0),
    (1 if Kicking_minus else 0),
    (1 if Gap_up_pg else 0),
    (1 if Exciting else 0),
    (1 if Boring else 0),
    (1 if morning_star  else 0),
    (1 if shooting_Star_bullish else 0),
    (1 if Bullish_Harami else 0),
    (1 if Bullish_Engulfing else 0),
    (1 if bullish_reversal else 0),
    (1 if Piercing_Line_bullish else 0),
    (1 if Hanging_Man_bullish else 0),
    (1 if evening_star else 0),
    (1 if shooting_Star_bearish else 0),
    (1 if bearish_harami else 0),
    (1 if Bearish_Engulfing else 0),
    (1 if bearish_reversal else 0),
    (1 if Hanging_Man_bearish else 0),

    (1 if Three_white_crows else 0),
    (1 if Three_Marubozu else 0),
    (1 if Three_black_crows else 0)
    
    ]
     
    
    

    if Green:
        strCandle=strCandle+'/ '+'Green'
        candle_score=candle_score+0

    if Red:
        strCandle=strCandle+'/ '+'Red'

    if Red_1:
        strCandle=strCandle+'/ '+'Red_1'

    if Gap_up_15_0:
        strCandle=strCandle+'/ '+'Gap_up_15'
        candle_score=candle_score+0

    if Gap_dn_15_0:
        strCandle=strCandle+'/ '+'Gap_dn_15'
        candle_score=candle_score-0
        
    if doji:
        strCandle=strCandle+'/ '+'doji'
        candle_score=candle_score+0

    if Exciting:
        strCandle=strCandle+'/ '+'Exciting'
        candle_score=candle_score+0

    if Boring:
        strCandle=strCandle+'/ '+'Boring'
        candle_score=candle_score-0

    if strength:
        strCandle=strCandle+'/ '+'strength'

    if Abandoned_baby:
        strCandle=strCandle+'/ '+'Abandoned_baby'
    
    '''
    
    '''
        
    if Kicking_plus:
        strCandle=strCandle+'/ '+'Kicking_plus'
        candle_score=candle_score+100
        
    if Kicking_plus_van:
        strCandle=strCandle+'/ '+'Kicking_plus_van'
        candle_score=candle_score+10

    if morning_star:
        strCandle=strCandle+'/ '+'morning_star'
        candle_score=candle_score+1
    if shooting_Star_bullish:
        strCandle=strCandle+'/ '+'shooting_Star_bullish'
        candle_score=candle_score+1
    if    Bullish_Harami:
        strCandle=strCandle+'/ '+'Bullish_Harami'
        candle_score=candle_score+1
    if    Bullish_Engulfing:
        strCandle=strCandle+'/ '+'Bullish_Engulfing'
        candle_score=candle_score+1
    if    bullish_reversal:
        strCandle=strCandle+'/ '+'bullish_reversal'
        candle_score=candle_score+1
    if    Piercing_Line_bullish:
        strCandle=strCandle+'/ '+'Piercing_Line_bullish'
        candle_score=candle_score+1
    if    Hanging_Man_bullish:
        strCandle=strCandle+'/ '+'Hanging_Man_bullish'
        candle_score=candle_score+1

    if    Three_Marubozu:
        strCandle=strCandle+'/ '+'Three_Marubozu'
        candle_score=candle_score
    if    Three_white_crows:
        strCandle=strCandle+'/ '+'Three_white_crows'
        candle_score=candle_score+1

    if    Three_black_crows:
        strCandle=strCandle+'/ '+'Three_black_crows'
        candle_score=candle_score-1


    if Kicking_minus:
        strCandle=strCandle+'/ '+'Kicking_minus'
        candle_score=candle_score-1
    if evening_star:
        strCandle=strCandle+'/ '+'evening_star'
        candle_score=candle_score-1
    if shooting_Star_bearish:
        strCandle=strCandle+'/ '+'shooting_Star_bearish'
        candle_score=candle_score-1
    if    hammer:
        strCandle=strCandle+'/ '+'hammer'
    if    inverted_hammer:
        strCandle=strCandle+'/ '+'inverted_hammer'
    if    bearish_harami:
        strCandle=strCandle+'/ '+'bearish_harami'
        candle_score=candle_score-1
    if    Bearish_Engulfing:
        strCandle=strCandle+'/ '+'Bearish_Engulfing'
        candle_score=candle_score-1
    if    bearish_reversal:
        strCandle=strCandle+'/ '+'bearish_reversal'
        candle_score=candle_score-1
    if    Hanging_Man_bearish:
        strCandle=strCandle+'/ '+'Hanging_Man_bearish'
        candle_score=candle_score-1
            
    #return candle_score
    #return candle_score,strCandle
    return lst_cols,lst_vals

def single_candle_score(lst_0,lst_1):    
    
    O_0,H_0,L_0,C_0=lst_0[0],lst_0[1],lst_0[2],lst_0[3]
    O_1,H_1,L_1,C_1=lst_1[0],lst_1[1],lst_1[2],lst_1[3]
    DojiSize = 0.1
    
    doji=(abs(O_0 - C_0) <= (H_0 - L_0) * DojiSize)
    doji_1=(abs(O_1 - C_1) <= (H_1 - L_1) * DojiSize)
    
    strength=((C_0 - O_0) <= ((H_0 - L_0) * 0.5))
    
    hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))
    
    inverted_hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((H_0 - C_0)/(.001 + H_0 - L_0) > 0.6) & ((H_0 - O_0)/(.001 + H_0 - L_0) > 0.6))
        
    Green=(C_0 > O_0)
    Green_1=(C_1 > O_1)
    
    Red=(C_0 <= O_0)
    Red_1=(C_1 <= O_1)

    Gap_up_15_0=(((C_0-C_1)*100/C_0)>1.5)&(((O_0-O_1)*100/O_0)>1.5)

    Gap_dn_15_0=(((C_0-C_1)*100/C_0)<-1.5)&(((O_0-O_1)*100/O_0)<-1.5)    
    
    #Abandoned_baby=doji_1&Gap_dn_15_1&Gap_up_15_0&Green
    MarubozuSize_50=0.50
    Marubozu_50_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize_50)
    Marubozu_50_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize_50)

    MarubozuSize_75=0.75
    Marubozu_75_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize_75)
    Marubozu_75_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize_75)
    
    
    MarubozuSize_100=0.98
    Marubozu_100_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize_100)
    Marubozu_100_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize_100)
    
    
    Marubozu_0=Marubozu_50_0
    Marubozu_1=Marubozu_50_1

    Marubozu=Marubozu_0
    
    gap_up_size=0.25
    gap_dn_size=-0.25

    Gap_up_95=(((C_0-C_1)*100/C_0)>gap_up_size)&(((O_0-O_1)*100/O_0)>gap_up_size)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<gap_dn_size)&(((O_0-O_1)*100/O_0)<gap_dn_size)

    Red_1=(C_1 < O_1)
    Kicking_plus_van=Green&Red_1&Gap_up_95&Marubozu_0&Marubozu_1
    Kicking_plus=Gap_up_95&Marubozu_0&Marubozu_1

    Kicking_minus=Red&Green_1&Gap_dn_95&Marubozu_0&Marubozu_1

    gap_up_size_pg=0.25
    Gap_up_pg=(((C_0-C_1)*100/C_0)>gap_up_size_pg)&(((O_0-O_1)*100/O_0)>gap_up_size_pg)&(((O_0-C_1)*100/O_0)>gap_up_size_pg)&(((C_0-O_1)*100/O_1)>gap_up_size_pg)
    
    body_ratio=abs(O_0 - C_0)/(H_0 - L_0)

    Exciting=(body_ratio>= 0.5)
    

    Boring=(body_ratio<0.5)
    strCandle=''
    candle_score=0
    
    lst_cols=['f_doji',
    'f_strength',
    'f_hammer',
    'f_inverted_hammer',
    'f_Green',
    'f_Gap_up_15_0',
    'f_Gap_dn_15_0',
    'f_Marubozu',
    'f_Gap_up_95',
    'f_Gap_dn_95',
    'f_Kicking_plus_van',
    'f_Kicking_plus',
    'f_Kicking_minus',
    'f_Gap_up_pg',
    'f_Exciting',
    'f_Boring']
    
    lst_vals=[(1 if doji else 0),
    (1 if strength else 0),
    (1 if hammer else 0),
    (1 if inverted_hammer else 0),
    (1 if Green else 0),
    (1 if Gap_up_15_0 else 0),
    (1 if Gap_dn_15_0 else 0),
    (1 if Marubozu else 0),
    (1 if Gap_up_95 else 0),
    (1 if Gap_dn_95 else 0),
    (1 if Kicking_plus_van else 0),
    (1 if Kicking_plus else 0),
    (1 if Kicking_minus else 0),
    (1 if Gap_up_pg else 0),
    (1 if Exciting else 0),
    (1 if Boring else 0)]
    
    if Green:
        strCandle=strCandle+'/ '+'Green'
        candle_score=candle_score+0

    if Red:
        strCandle=strCandle+'/ '+'Red'

    if Gap_up_15_0:
        strCandle=strCandle+'/ '+'Gap_up_15'
        candle_score=candle_score+0

    if Gap_dn_15_0:
        strCandle=strCandle+'/ '+'Gap_dn_15'
        candle_score=candle_score-0
        
    if doji:
        strCandle=strCandle+'/ '+'doji'
        candle_score=candle_score+0

    if Exciting:
        strCandle=strCandle+'/ '+'Exciting'
        candle_score=candle_score+0

    if Boring:
        strCandle=strCandle+'/ '+'Boring'
        candle_score=candle_score-0

    if strength:
        strCandle=strCandle+'/ '+'strength'

    if Kicking_plus:
        strCandle=strCandle+'/ '+'Kicking_plus'
        candle_score=candle_score+100
        
    if Kicking_plus_van:
        strCandle=strCandle+'/ '+'Kicking_plus_van'
        candle_score=candle_score+10

    if Kicking_minus:
        strCandle=strCandle+'/ '+'Kicking_minus'
        candle_score=candle_score-1

        
            
    #return candle_score
    #return candle_score,strCandle
    return lst_cols,lst_vals

def candle_score3(lst_0,lst_1,lst_2):    
    
    O_0,H_0,L_0,C_0=lst_0[0],lst_0[1],lst_0[2],lst_0[3]
    O_1,H_1,L_1,C_1=lst_1[0],lst_1[1],lst_1[2],lst_1[3]
    O_2,H_2,L_2,C_2=lst_2[0],lst_2[1],lst_2[2],lst_2[3]
    
    DojiSize = 0.1
    
    doji=(abs(O_0 - C_0) <= (H_0 - L_0) * DojiSize)
    
    doji_1=(abs(O_1 - C_1) <= (H_1 - L_1) * DojiSize)
    
    
    strength=((C_0 - O_0) <= ((H_0 - L_0) * 0.5))
    
    hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((C_0 - L_0)/(.001 + H_0 - L_0) > 0.6) & ((O_0 - L_0)/(.001 + H_0 - L_0) > 0.6))
    
    inverted_hammer=(((H_0 - L_0)>3*(O_0 -C_0)) &  ((H_0 - C_0)/(.001 + H_0 - L_0) > 0.6) & ((H_0 - O_0)/(.001 + H_0 - L_0) > 0.6))
    
    bullish_reversal= (O_2 > C_2)&(O_1 > C_1)&doji
    
    bearish_reversal= (O_2 < C_2)&(O_1 < C_1)&doji
    
    evening_star=(C_2 > O_2) & (min(O_1, C_1) > C_2) & (O_0 < min(O_1, C_1)) & (C_0 < O_0 )
    
    morning_star=(C_2 < O_2) & (min(O_1, C_1) < C_2) & (O_0 > min(O_1, C_1)) & (C_0 > O_0 )
    
    shooting_Star_bearish=(O_1 < C_1) & (O_0 > C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    shooting_Star_bullish=(O_1 > C_1) & (O_0 < C_1) & ((H_0 - max(O_0, C_0)) >= abs(O_0 - C_0) * 3) & ((min(C_0, O_0) - L_0 )<= abs(O_0 - C_0)) & inverted_hammer
    
    bearish_harami=(C_1 > O_1) & (O_0 > C_0) & (O_0 <= C_1) & (O_1 <= C_0) & ((O_0 - C_0) < (C_1 - O_1 ))
    
    Bullish_Harami=(O_1 > C_1) & (C_0 > O_0) & (C_0 <= O_1) & (C_1 <= O_0) & ((C_0 - O_0) < (O_1 - C_1))
    
    Bearish_Engulfing=((C_1 > O_1) & (O_0 > C_0)) & ((O_0 >= C_1) & (O_1 >= C_0)) & ((O_0 - C_0) > (C_1 - O_1 ))
    
    Bullish_Engulfing=(O_1 > C_1) & (C_0 > O_0) & (C_0 >= O_1) & (C_1 >= O_0) & ((C_0 - O_0) > (O_1 - C_1 ))
    
    Piercing_Line_bullish=(C_1 < O_1) & (C_0 > O_0) & (O_0 < L_1) & (C_0 > C_1)& (C_0>((O_1 + C_1)/2)) & (C_0 < O_1)

    Hanging_Man_bullish=(C_1 < O_1) & (O_0 < L_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Hanging_Man_bearish=(C_1 > O_1) & (C_0>((O_1 + C_1)/2)) & (C_0 < O_1) & hammer

    Green=(C_0 > O_0)
    Green_1=(C_1 > O_1)
    
    Red=(C_0 <= O_0)

    Gap_up_15_0=(((C_0-C_1)*100/C_0)>1.5)&(((O_0-O_1)*100/O_0)>1.5)

    Gap_dn_15_0=(((C_0-C_1)*100/C_0)<-1.5)&(((O_0-O_1)*100/O_0)<-1.5)


    Gap_up_15_1=(((C_1-C_2)*100/C_1)>1.5)&(((O_1-O_2)*100/O_1)>1.5)

    Gap_dn_15_1=(((C_1-C_2)*100/C_1)<-1.5)&(((O_1-O_2)*100/O_1)<-1.5)
    
    
    
    Abandoned_baby=doji_1&Gap_dn_15_1&Gap_up_15_0&Green
    MarubozuSize=0.50
    Marubozu_0=(abs(O_0 - C_0) >= (H_0 - L_0) * MarubozuSize)
    Marubozu_1=(abs(O_1 - C_1) >= (H_1 - L_1) * MarubozuSize)
    
    gap_up_size=0.35
    gap_dn_size=-0.25

    Gap_up_95=(((C_0-C_1)*100/C_0)>gap_up_size)&(((O_0-O_1)*100/O_0)>gap_up_size)
    Gap_dn_95=(((C_0-C_1)*100/C_0)<gap_dn_size)&(((O_0-O_1)*100/O_0)<gap_dn_size)

    Red_1=(C_1 < O_1)
    Kicking_plus_van=Green&Red_1&Gap_up_95&Marubozu_0&Marubozu_1
    Kicking_plus=Gap_up_95&Marubozu_0&Marubozu_1

    Kicking_minus=Red&Green_1&Gap_dn_95&Marubozu_0&Marubozu_1

    
    body_ratio0=abs(O_0 - C_0)/(H_0 - L_0)
    body_ratio1=abs(O_1 - C_1)/(H_1 - L_1)
    body_ratio2=abs(O_1 - C_2)/(H_2 - L_2)

    Exciting0=(body_ratio0>= 0.5)
    
    Exciting1=(body_ratio1>= 0.5)
    Exciting2=(body_ratio2>= 0.5)
    
    Exciting=Exciting0|Exciting1|Exciting2
    Exciting=Exciting0

    Boring0=(body_ratio0<0.5)
    Boring1=(body_ratio1<0.5)
    Boring2=(body_ratio2<0.5)
    
    Boring=Boring0|Boring1|Boring2
    Boring=Boring0
    
    strCandle=''
    candle_score=0



    if Green:
        strCandle=strCandle+'/ '+'Green'
        candle_score=candle_score+0

    if Red:
        strCandle=strCandle+'/ '+'Red'

    if Red_1:
        strCandle=strCandle+'/ '+'Red_1'

    if Gap_up_15_0:
        strCandle=strCandle+'/ '+'Gap_up_15'
        candle_score=candle_score+0

    if Gap_dn_15_0:
        strCandle=strCandle+'/ '+'Gap_dn_15'
        candle_score=candle_score-0
        
    if doji:
        strCandle=strCandle+'/ '+'doji'
        candle_score=candle_score+0

    if Exciting:
        strCandle=strCandle+'/ '+'Exciting'
        candle_score=candle_score+0

    if Boring:
        strCandle=strCandle+'/ '+'Boring'
        candle_score=candle_score-0

    if strength:
        strCandle=strCandle+'/ '+'strength'

    if Abandoned_baby:
        strCandle=strCandle+'/ '+'Abandoned_baby'
        
        
    if Kicking_plus:
        strCandle=strCandle+'/ '+'Kicking_plus'
        candle_score=candle_score+100
        
    if Kicking_plus_van:
        strCandle=strCandle+'/ '+'Kicking_plus_van'
        candle_score=candle_score+10

    if morning_star:
        strCandle=strCandle+'/ '+'morning_star'
        candle_score=candle_score+1
    if shooting_Star_bullish:
        strCandle=strCandle+'/ '+'shooting_Star_bullish'
        candle_score=candle_score+1
    if    Bullish_Harami:
        strCandle=strCandle+'/ '+'Bullish_Harami'
        candle_score=candle_score+1
    if    Bullish_Engulfing:
        strCandle=strCandle+'/ '+'Bullish_Engulfing'
        candle_score=candle_score+1
    if    bullish_reversal:
        strCandle=strCandle+'/ '+'bullish_reversal'
        candle_score=candle_score+1
    if    Piercing_Line_bullish:
        strCandle=strCandle+'/ '+'Piercing_Line_bullish'
        candle_score=candle_score+1
    if    Hanging_Man_bullish:
        strCandle=strCandle+'/ '+'Hanging_Man_bullish'
        candle_score=candle_score+1
    if Kicking_minus:
        strCandle=strCandle+'/ '+'Kicking_minus'
        candle_score=candle_score-1

    
    if evening_star:
        strCandle=strCandle+'/ '+'evening_star'
        candle_score=candle_score-1

    if shooting_Star_bearish:
        strCandle=strCandle+'/ '+'shooting_Star_bearish'
        candle_score=candle_score-1
    if    hammer:
        strCandle=strCandle+'/ '+'hammer'
    if    inverted_hammer:
        strCandle=strCandle+'/ '+'inverted_hammer'
    if    bearish_harami:
        strCandle=strCandle+'/ '+'bearish_harami'
        candle_score=candle_score-1
    if    Bearish_Engulfing:
        strCandle=strCandle+'/ '+'Bearish_Engulfing'
        candle_score=candle_score-1
    if    bearish_reversal:
        strCandle=strCandle+'/ '+'bearish_reversal'
        candle_score=candle_score-1
    if    Hanging_Man_bearish:
        strCandle=strCandle+'/ '+'Hanging_Man_bearish'
        candle_score=candle_score-1
            
    #return candle_score
    return candle_score,strCandle

'''
def candle_df(df):
    df_candle=first_letter_upper(df)
    
    df_candle['candle_score']=0
    
    for c in range(2,len(df_candle)):
        lst_2=[df_candle['Open'].iloc[c-2],df_candle['High'].iloc[c-2],df_candle['Low'].iloc[c-2],df_candle['Close'].iloc[c-2]]
        lst_1=[df_candle['Open'].iloc[c-1],df_candle['High'].iloc[c-1],df_candle['Low'].iloc[c-1],df_candle['Close'].iloc[c-1]]
        lst_0=[df_candle['Open'].iloc[c],df_candle['High'].iloc[c],df_candle['Low'].iloc[c],df_candle['Close'].iloc[c]]
        cscore=candle_score(lst_0,lst_1,lst_2)    
        df_candle['candle_score'].iat[c]=cscore
    
    df_candle['candle_cumsum']=df_candle['candle_score'].rolling(3).sum()
    
    return df_candle
'''
def candle_df(df):
    #df_candle=first_letter_upper(df)
    df_candle=df.copy()
    df_candle['candle_score']=0
    df_candle['candle_pattern']=''
    
    for c in range(2,len(df_candle)):
        cscore,cpattern=0,''
        lst_2=[df_candle['Open'].iloc[c-2],df_candle['High'].iloc[c-2],df_candle['Low'].iloc[c-2],df_candle['Close'].iloc[c-2]]
        lst_1=[df_candle['Open'].iloc[c-1],df_candle['High'].iloc[c-1],df_candle['Low'].iloc[c-1],df_candle['Close'].iloc[c-1]]
        lst_0=[df_candle['Open'].iloc[c],df_candle['High'].iloc[c],df_candle['Low'].iloc[c],df_candle['Close'].iloc[c]]
        cscore,cpattern=candle_score(lst_0,lst_1,lst_2)    
        df_candle['candle_score'].iat[c]=cscore
        df_candle['candle_pattern'].iat[c]=str(cpattern)
    
    df_candle['candle_cumsum']=df_candle['candle_score'].rolling(3).sum()
    
    return df_candle

def single_candle_df(df):
    #df_candle=first_letter_upper(df)
    df_candle=df.copy()
    df_candle['candle_score']=0
    df_candle['candle_pattern']=''
    
    lst_cols=['f_doji',
    'f_strength',
    'f_hammer',
    'f_inverted_hammer',
    'f_Green',
    'f_Gap_up_15_0',
    'f_Gap_dn_15_0',
    'f_Marubozu',
    'f_Gap_up_95',
    'f_Gap_dn_95',
    'f_Kicking_plus_van',
    'f_Kicking_plus',
    'f_Kicking_minus',
    'f_Gap_up_pg',
    'f_Exciting',
    'f_Boring',
    
    'f_morning_star',
    'f_shooting_Star_bullish',
    'f_Bullish_Harami',
    'f_Bullish_Engulfing',
    'f_bullish_reversal',
    'f_Piercing_Line_bullish',
    'f_Hanging_Man_bullish',
    'f_evening_star',
    'f_shooting_Star_bearish',
    'f_bearish_harami',
    'f_Bearish_Engulfing',
    'f_bearish_reversal',
    'f_Hanging_Man_bearish',
    'f_Three_white_crows',
    'f_Three_Marubozu',
    'f_Three_black_crows']


    
    for col in lst_cols:
        df_candle[col]=0
        
    for c in range(2,len(df_candle)-1):
        cscore,cpattern=0,''
        lst_2=[df_candle['Open'].iloc[c-2],df_candle['High'].iloc[c-2],df_candle['Low'].iloc[c-2],df_candle['Close'].iloc[c-2]]

        lst_1=[df_candle['Open'].iloc[c-1],df_candle['High'].iloc[c-1],df_candle['Low'].iloc[c-1],df_candle['Close'].iloc[c-1]]
        lst_0=[df_candle['Open'].iloc[c],df_candle['High'].iloc[c],df_candle['Low'].iloc[c],df_candle['Close'].iloc[c]]
        lst_col,lst_val=candle_score_v4(lst_0,lst_1,lst_2) 
        x=0
        for col in lst_cols:
            print(x, col, lst_val,x)
            df_candle[col].iat[c]=lst_val[x]
            x=x+1
    
    df_candle['candle_cumsum']=df_candle['candle_score'].rolling(3).sum()
    
    return df_candle

def candle_df2(df,type='bull'):
    #df_candle=first_letter_upper(df)
    df_candle=df.copy()
    df_candle['candle_score']=0
    df_candle['candle_pattern']=''
    
    for c in range(2,len(df_candle)):
        cscore,cpattern=0,''
        lst_2=[df_candle['Open'].iloc[c-2],df_candle['High'].iloc[c-2],df_candle['Low'].iloc[c-2],df_candle['Close'].iloc[c-2]]
        lst_1=[df_candle['Open'].iloc[c-1],df_candle['High'].iloc[c-1],df_candle['Low'].iloc[c-1],df_candle['Close'].iloc[c-1]]
        lst_0=[df_candle['Open'].iloc[c],df_candle['High'].iloc[c],df_candle['Low'].iloc[c],df_candle['Close'].iloc[c]]
        cscore,cpattern=candle_score2(lst_0,lst_1,lst_2,type)    
        df_candle['candle_score'].iat[c]=cscore
        df_candle['candle_pattern'].iat[c]=cpattern
    
    df_candle['candle_cumsum']=df_candle['candle_score'].rolling(3).sum()
    
    return df_candle

def stochastic(df,calc_period,smoothening,smoothening2):
    
    df=first_letter_upper(df)
    
    df['L14']=df['Low'].rolling(calc_period).min()
    df['H14']=df['High'].rolling(calc_period).max()
    
    df['%K_Fast']=((df['Close']-df['L14'])/(df['H14']-df['L14']))*100
    
    df['%D_Fast']=df['%K_Fast'].rolling(smoothening).mean()
    
    df['%K_Slow']=df['%K_Fast'].rolling(smoothening2).mean()
    
    df['%D_Slow']=df['%K_Slow'].rolling(smoothening2).mean()
    
        
    return df


def HA_candle_score(df):
    
    df=first_letter_upper(df)

    
    #df=HA(df, ohlc=['Open', 'High', 'Low', 'Close'])
    
    #df=df[['Open','High','Low','Close','HA_Open','HA_High','HA_Low','HA_Close']].copy()
    
    bullish_no_wick=((df['HA_Open']-df['HA_Low'])*100/df['HA_Open']<0.1)
    
    bearish_no_wick=((df['HA_High']-df['HA_Open'])*100/df['HA_Open']<0.1)
    
    DojiSize = 0.1
    
    doji=(abs(df['HA_Open'] - df['HA_Close']) <= (df['HA_High'] - df['HA_Low']) * DojiSize)
    
    doji1=(abs(df['HA_Open'].shift(1) - df['HA_Close'].shift(1)) <= (df['HA_High'].shift(1) - df['HA_Low'].shift(1)) * DojiSize)
    
    
    bullish_reversal_doji= (df['HA_Open'].shift(3) > df['HA_Close'].shift(3))&(df['HA_Open'].shift(2) > df['HA_Close'].shift(2))&doji1&(df['HA_Open'] < df['HA_Close'])
    
    bearish_reversal_doji= (df['HA_Open'].shift(3) < df['HA_Close'].shift(3))&(df['HA_Open'].shift(2) < df['HA_Close'].shift(2))&doji1&(df['HA_Open'] > df['HA_Close'])
    
    #df['bullish_wick']=(df['HA_Open']-df['HA_Low'])*100/df['HA_Open']
    #df['bearish_wick']=(df['HA_High']-df['HA_Open'])*100/df['HA_Open']
    
    bullish_reversal_wick= (df['HA_Open'].shift(2) > df['HA_Close'].shift(2))&(df['HA_Open'].shift(1) > df['HA_Close'].shift(1)) & ((df['HA_Open']-df['HA_Low'])*100/df['HA_Open']<0.4) & (df['HA_Open'] < df['HA_Close'])
    
    bearish_reversal_wick= (df['HA_Open'].shift(2) < df['HA_Close'].shift(2))&(df['HA_Open'].shift(1) < df['HA_Close'].shift(1)) & ((df['HA_High']-df['HA_Open'])*100/df['HA_Open']<0.4) & (df['HA_Open'] > df['HA_Close'])
    
    doji2=(abs(df['HA_Open'].shift(2) - df['HA_Close'].shift(2)) <= (df['HA_High'].shift(2) - df['HA_Low'].shift(2)) * DojiSize)
    
    bullish_reversal_doji2= doji2& (df['HA_Open'].shift(1) < df['HA_Close'].shift(1)) & (df['HA_Open'] < df['HA_Close'])  

    bearish_reversal_doji2= doji2& (df['HA_Open'].shift(1) > df['HA_Close'].shift(1)) & (df['HA_Open'] > df['HA_Close'])  
    '''
    bullish_reversal_double= (df['HA_Open'].shift(3) > df['HA_Close'].shift(3)) & (df['HA_Open'].shift(2) > df['HA_Close'].shift(2)) &(df['HA_Open'].shift(1) < df['HA_Close'].shift(1)) & (df['HA_Open'] < df['HA_Close'])  
    
    bearish_reversal_double= (df['HA_Open'].shift(3) < df['HA_Close'].shift(3)) & (df['HA_Open'].shift(2) < df['HA_Close'].shift(2)) &(df['HA_Open'].shift(1) > df['HA_Close'].shift(1)) & (df['HA_Open'] > df['HA_Close'])  
    '''
    bullish_reversal_double=  (df['HA_Open'].shift(2) > df['HA_Close'].shift(2)) &(df['HA_Open'].shift(1) < df['HA_Close'].shift(1)) & (df['HA_Open'] < df['HA_Close'])      
    bearish_reversal_double=  (df['HA_Open'].shift(2) < df['HA_Close'].shift(2)) &(df['HA_Open'].shift(1) > df['HA_Close'].shift(1)) & (df['HA_Open'] > df['HA_Close'])  

    bullish_reversal_single=  (df['HA_Open'].shift(1) > df['HA_Close'].shift(1)) & (df['HA_Open'] < df['HA_Close'])      
    bearish_reversal_single=  (df['HA_Open'].shift(1) < df['HA_Close'].shift(1)) & (df['HA_Open'] > df['HA_Close'])  

        
    df['bullish_reversal_doji']=np.where(bullish_reversal_doji,1,0)
    df['bullish_reversal_doji2']=np.where(bullish_reversal_doji2,1,0)
    df['bullish_reversal_wick']=np.where(bullish_reversal_wick,1,0)
    df['bullish_reversal_double']=np.where(bullish_reversal_double,1,0)
    df['bullish_reversal_single']=np.where(bullish_reversal_single,1,0)
    
    df['bearish_reversal_doji']=np.where(bearish_reversal_doji,1,0)
    df['bearish_reversal_doji2']=np.where(bearish_reversal_doji2,1,0)
    df['bearish_reversal_wick']=np.where(bearish_reversal_wick,1,0)
    df['bearish_reversal_double']=np.where(bearish_reversal_double,1,0)
    df['bearish_reversal_single']=np.where(bearish_reversal_single,1,0)

    
    #df['bearish']=df['bearish_reversal_doji']+df['bearish_reversal_doji2']+df['bearish_reversal_wick']+df['bearish_reversal_double']
    #df['bullish']=df['bullish_reversal_doji']+df['bullish_reversal_doji2']+df['bullish_reversal_wick']+df['bullish_reversal_double']

    df['bearish']=df['bearish_reversal_single']+df['bearish_reversal_doji2']+df['bearish_reversal_wick']+df['bearish_reversal_double']
    df['bullish']=df['bullish_reversal_single']+df['bullish_reversal_doji2']+df['bullish_reversal_wick']+df['bullish_reversal_double']
    
    df['HA_candle_score']=df['bullish']-df['bearish']


    return df

def pivotpoint_df(df):
    df=first_letter_upper(df)
    df['PP']=round((df['Close'].shift(1)+df['High'].shift(1)+df['Low'].shift(1))/3,2)
    df['S1']=df['PP']*2-df['High'].shift(1)
    df['S2']=df['PP']-(df['High'].shift(1)-df['Low'].shift(1))
    
    df['R1']=df['PP']*2-df['Low'].shift(1)
    df['R2']=df['PP']+(df['High'].shift(1)-df['Low'].shift(1))

    df['R3']=df['R1']+(df['High'].shift(1)-df['Low'].shift(1))
    df['S3']=df['S1']-(df['High'].shift(1)-df['Low'].shift(1))
    
    
    df['BC']=round((df['High'].shift(1)+df['Low'].shift(1))/2,2)

    df['TC'] = (df['PP']-df['BC'])+ df['PP']

    
    return df


def pivotpoint(PREV_HIGH,PREV_LOW,PREV_CLOSE):
    
    PP=round((PREV_HIGH+PREV_LOW+PREV_CLOSE)/3,2)
    S1=PP*2-PREV_HIGH
    S2=PP-(PREV_HIGH-PREV_LOW)

    R1=PP*2-PREV_LOW
    R2=PP+(PREV_HIGH-PREV_LOW)
    
    return PP,S1,S2,R1,R2


def ema_crossover_3(df,long1,long2,short,price_field="Close"):
    
    df=first_letter_upper(df)
    
    df['Ewm_long1'],df['Ewm_long2'],df['Ewm_short']=df[price_field].ewm(span=long1,adjust=False).mean(), df[price_field].ewm(span=long2,adjust=False).mean(),df[price_field].ewm(span=short,adjust=False).mean()
    
    return df

def ema_crossover(df,long,short,price_field="Close"):
    
    df=first_letter_upper(df)
    
    df['Ewm_long'],df['Ewm_short']=df[price_field].ewm(span=long,adjust=False).mean(), df[price_field].ewm(span=short,adjust=False).mean()
    
    return df

def CCI(data, ndays): 
    data['TP'] = (data['High'] + data['Low'] + data['Close']) / 3 
    data['TP_mean']=data['TP'].rolling(ndays).mean()
    data['TP_std']=data['TP'].rolling(ndays).std()
    data['CCI_'+str(ndays)] = (data['TP']-data['TP_mean']) / (0.015 * data['TP_std']) 
    data['CCI'] = (data['TP']-data['TP_mean']) / (0.015 * data['TP_std']) 

    return data

def DirectionPoints(dfSeries, minSegSize=1, sizeInDevs=0.5):
    minRetrace = minSegSize
    
    curVal = dfSeries[0]
    curPos = dfSeries.index[0]
    curDir = 1
    #dfRes = pd.DataFrame(np.zeros((len(dfSeries.index), 2)), index=dfSeries.index, columns=["Dir", "Value"])
    dfRes = pd.DataFrame(index=dfSeries.index, columns=["Dir", "Value"])
    #print(dfRes)
    #print(len(dfSeries.index))
    for ln in dfSeries.index:
        if((dfSeries[ln] - curVal)*curDir >= 0):
            curVal = dfSeries[ln]
            curPos = ln
            #print(str(ln) + ": moving curVal further, to " + str(curVal))
        else:      
            retracePrc = abs((dfSeries[ln]-curVal)/curVal*100)
            #print(str(ln) + ": estimating retracePrc, it's " + str(retracePrc))
            if(retracePrc >= minRetrace):
                #print(str(ln) + ": registering key point, its pos is " + str(curPos) + ", value = " + str(curVal) + ", dir=" +str(curDir))
                dfRes.at[curPos, 'Value'] = curVal
                dfRes.at[curPos, 'Dir'] = curDir
                curVal = dfSeries[ln]
                curPos = ln
                curDir = -1*curDir
                #print(str(ln) + ": setting new cur vals, pos is " + str(curPos) + ", curVal = " + str(curVal) + ", dir=" +str(curDir))
        #print(ln, curVal, curDir)
    dfRes[['Value']] = dfRes[['Value']].astype(float)
    dfRes = dfRes.interpolate(method='linear')
    return(dfRes)
