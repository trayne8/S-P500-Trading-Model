import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import coint

sns.set_style('whitegrid')


@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(tickers, (list, tuple)):
        close = df['Close'].copy()
    else:
        close = df['Close'].to_frame()
    close.index = pd.to_datetime(close.index)
    return close.dropna()


def train_test_split(df, train_frac=0.7):
    split = int(len(df) * train_frac)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def fit_hedge_ratio(y, x):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model, float(model.params[1])


def regression_metrics(model, y, x):
    preds = model.predict(sm.add_constant(x))
    resid = y - preds
    rmse = np.sqrt((resid ** 2).mean())
    return {
        'RMSE': float(rmse),
        'Adj. R-squared': float(model.rsquared_adj),
        'Durbin-Watson': float(durbin_watson(resid)),
    }


def generate_signals(df, beta, lookback_mean=60, lookback_std=60, enter_z=2.0, exit_z=0.5):
    spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
    rm = spread.rolling(lookback_mean).mean()
    rs = spread.rolling(lookback_std).std()
    z = (spread - rm) / rs

    sig = pd.Series(0, index=df.index)
    sig[z < -enter_z] = 1
    sig[z > enter_z] = -1
    sig[np.abs(z) < exit_z] = 0
    pos = sig.replace(0, np.nan).ffill().fillna(0).astype(int)

    return pd.DataFrame({'spread': spread, 'z': z, 'signal_raw': sig, 'position': pos})


def compute_strategy_returns(df, beta, positions, cost_rate=0.0, slippage_per_share=0.0):
    px = df.copy()
    px_ret = px.diff()
    pos_sp = positions
    pos_other = -beta * positions
    pnl = pos_sp.shift(1) * px_ret.iloc[:, 0] + pos_other.shift(1) * px_ret.iloc[:, 1]
    pnl = pnl.fillna(0)

    trades_sp = pos_sp.diff().abs().fillna(0)
    trades_other = pos_other.diff().abs().fillna(0)
    trade_value = trades_sp * px.iloc[:, 0] + trades_other * px.iloc[:, 1]
    cost = trade_value * cost_rate
    slippage_cost = (trades_sp + trades_other) * slippage_per_share

    net = pnl - cost - slippage_cost
    initial_notional = float(px.iloc[0, 0])
    strategy_ret = net / initial_notional
    cumret = (1 + strategy_ret).cumprod() - 1
    return strategy_ret, cumret


def sharpe_ratio(returns, periods=252):
    ann = returns.mean() * periods
    vol = returns.std() * np.sqrt(periods)
    return float(ann / vol) if vol != 0 else np.nan


def max_drawdown(cumret):
    running_max = cumret.cummax()
    return float(((cumret - running_max) / running_max).min())


def sensitivity_analysis(df, beta, positions, train_last_idx):
    cost_rates = [0.0, 0.0001, 0.0005, 0.001]
    slippages = [0.0, 0.01, 0.05]
    rows = []
    for cost in cost_rates:
        for slip in slippages:
            ret, cum = compute_strategy_returns(df, beta, positions, cost_rate=cost, slippage_per_share=slip)
            train_mask = ret.index <= train_last_idx
            test_mask = ret.index > train_last_idx
            rows.append({
                'cost_rate': cost,
                'slippage_per_share': slip,
                'train_sharpe': sharpe_ratio(ret[train_mask]),
                'test_sharpe': sharpe_ratio(ret[test_mask]),
                'train_maxdd': max_drawdown(cum[train_mask]),
                'test_maxdd': max_drawdown(cum[test_mask]),
            })
    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title='Pairs Trading Dashboard', layout='wide')
    st.title('Pairs Trading Backtester')
    st.markdown(
        'Build and compare a pairs trading signal for two co-moving assets using Python, ' 
        'cointegration, OLS regression, and interactive Streamlit dashboards.'
    )

    with st.sidebar:
        st.header('Inputs')
        tickers = st.text_input('Tickers (comma-separated)', '^GSPC, ^IXIC')
        start_date = st.date_input('Start date', datetime.datetime(2018, 1, 1))
        end_date = st.date_input('End date', datetime.date.today())
        train_frac = st.slider('Train split', 0.5, 0.9, 0.7, 0.05)
        lookback = st.slider('Spread lookback days', 20, 120, 60, 5)
        enter_z = st.slider('Entry z-score', 1.0, 3.0, 2.0, 0.1)
        exit_z = st.slider('Exit z-score', 0.1, 1.0, 0.5, 0.1)
        cost_rate = st.number_input('Transaction cost rate', 0.0, 0.01, 0.0005, format='%.5f')
        slippage = st.number_input('Slippage per share', 0.0, 0.1, 0.01, format='%.2f')
        show_sensitivity = st.checkbox('Show cost/slippage sensitivity', value=True)

    ticker_list = [t.strip() for t in tickers.split(',') if t.strip()]
    if len(ticker_list) != 2:
        st.error('Please provide exactly two tickers, separated by a comma.')
        return

    if start_date >= end_date:
        st.error('Start date must be earlier than end date.')
        return

    data_load_state = st.info('Fetching historical data...')
    close = fetch_data(ticker_list, start_date, end_date)
    data_load_state.empty()

    if close.empty or close.shape[1] < 2:
        st.error('Could not load data for the requested tickers/dates.')
        return

    train, test = train_test_split(close, train_frac)
    base, partner = close.columns[0], close.columns[1]

    st.subheader('Price series')
    st.line_chart(close)

    st.subheader('Correlation and train/test split')
    corr = close.pct_change().corr()
    st.dataframe(corr.style.format('{:.4f}'))
    st.write('Train period:', train.index[0].date(), 'to', train.index[-1].date())
    st.write('Test period:', test.index[0].date(), 'to', test.index[-1].date())

    st.subheader('Exploratory scatter matrix (train returns)')
    train_returns = np.log(train).diff().dropna()
    grid = sns.pairplot(train_returns)
    st.pyplot(grid.fig)

    coint_t, pvalue, _ = coint(train[base], train[partner])
    model, beta = fit_hedge_ratio(train[base], train[partner])
    metrics = regression_metrics(model, train[base], train[partner])

    st.subheader('Model & cointegration')
    col1, col2 = st.columns(2)
    col1.metric('Hedge ratio β', f'{beta:.4f}')
    col1.metric('Cointegration p-value', f'{pvalue:.4f}')
    col2.metric('RMSE', f'{metrics["RMSE"]:.2f}')
    col2.metric('Adj. R²', f'{metrics["Adj. R-squared"]:.4f}')
    col2.metric('Durbin-Watson', f'{metrics["Durbin-Watson"]:.4f}')

    signals = generate_signals(close[[base, partner]], beta, lookback, lookback, enter_z, exit_z)
    strat_ret, strat_cum = compute_strategy_returns(close[[base, partner]], beta, signals['position'], cost_rate=cost_rate, slippage_per_share=slippage)

    train_mask = strat_ret.index <= train.index[-1]
    test_mask = strat_ret.index > train.index[-1]
    train_sh = sharpe_ratio(strat_ret[train_mask])
    test_sh = sharpe_ratio(strat_ret[test_mask])
    train_mdd = max_drawdown(strat_cum[train_mask])
    test_mdd = max_drawdown(strat_cum[test_mask])

    st.subheader('Strategy performance')
    perf_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'Sharpe': [train_sh, test_sh],
        'Max Drawdown': [train_mdd, test_mdd],
    })
    st.dataframe(perf_df.style.format({'Sharpe': '{:.4f}', 'Max Drawdown': '{:.4f}'}))

    st.line_chart(strat_cum.rename('Cumulative Return'))

    with st.expander('Show spread and z-score'): 
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        signals['spread'].plot(ax=ax[0], title='Spread')
        ax[0].axhline(signals['spread'].rolling(lookback).mean().iloc[-1], color='gray', linestyle='--')
        signals['z'].plot(ax=ax[1], title='Z-score')
        ax[1].axhline(enter_z, color='red', linestyle='--')
        ax[1].axhline(-enter_z, color='green', linestyle='--')
        ax[1].axhline(exit_z, color='gray', linestyle=':')
        ax[1].axhline(-exit_z, color='gray', linestyle=':')
        st.pyplot(fig)

    if show_sensitivity:
        st.subheader('Sensitivity to transaction costs and slippage')
        sensitivity = sensitivity_analysis(close[[base, partner]], beta, signals['position'], train.index[-1])
        st.dataframe(sensitivity.style.format({
            'cost_rate': '{:.5f}',
            'slippage_per_share': '{:.2f}',
            'train_sharpe': '{:.4f}',
            'test_sharpe': '{:.4f}',
            'train_maxdd': '{:.4f}',
            'test_maxdd': '{:.4f}'
        }))

    st.write('---')
    st.markdown('**Notes:** This dashboard is built in Python and rendered through Streamlit. It supports interactive parameter changes and updates charts in HTML automatically.')


if __name__ == '__main__':
    main()
