"""
通用多因子回测框架（40+因子完整版）
股票池：沪深300/中证500/中证800成分股
结果保存：res/{因子名}/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os
from pathlib import Path

# ==================== 基础配置 ====================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

N_GROUPS = 5
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent

START_DATE = ""
END_DATE = ""

DATA_PATHS = {
    '沪深300': PROJECT_ROOT / 'data' / 'hs300_stocks_2024.csv',
    '中证500': PROJECT_ROOT / 'data' / 'zz500_stocks_2024.csv',
    '中证800': PROJECT_ROOT / 'data' / 'zz800_stocks_2024.csv'
}
RES_ROOT = PROJECT_ROOT / 'res'

# ==================== 因子函数字典（明确定义在全局） ====================
FACTORS = {
    # 情绪 & 量价
    'ARBR': lambda df: calc_arbr(df),
    'PSY12': lambda df: calc_psy(df, 12),
    'PVT': lambda df: calc_pvt(df),
    'SRDM30': lambda df: calc_srdm(df, 30),
    'SI': lambda df: calc_si(df),
    'MICD': lambda df: calc_micd(df),
    'VROC12': lambda df: calc_vroc(df, 12),
    'BOLL_MID': lambda df: calc_boll_mid(df),
    'OBV': lambda df: calc_obv(df),
    'WAD30': lambda df: calc_wad(df, 30),
    'BBIBOLL': lambda df: calc_bbiboll(df),
    'BBI': lambda df: calc_bbi(df),
    'MFI': lambda df: calc_mfi(df),
    'MA5': lambda df: calc_ma(df, 5),
    'PRICEOSC': lambda df: calc_priceosc(df),
    'EXPMA5': lambda df: calc_expma(df, 5),
    'CDP': lambda df: calc_cdp(df),
    'RSI': lambda df: calc_rsi(df),
    'ATR14': lambda df: calc_atr(df, 14),
    'DPO': lambda df: calc_dpo(df),
    'RCCD': lambda df: calc_rccd(df),
    'WR': lambda df: calc_wr(df),
    'ENV14': lambda df: calc_env(df, 14),
    'VMACD': lambda df: calc_vmacd(df),
    'WR12': lambda df: calc_wr12(df),
    'CVLT10': lambda df: calc_cvlt(df, 10),
    'CCI': lambda df: calc_cci(df),
    'VOLRATIO5': lambda df: calc_volratio(df, 5),
    'KDJ': lambda df: calc_kdj(df),
    'LWR': lambda df: calc_lwr(df),
    'DMA': lambda df: calc_dma(df),
    'ADTM': lambda df: calc_adtm(df),
    'SRMI9': lambda df: calc_srmi(df, 9),
    'BIAS': lambda df: calc_bias(df),
    'VSTD10': lambda df: calc_vstd(df, 10),
    'MACD': lambda df: calc_macd(df),
    'VRSI6': lambda df: calc_vrsi(df, 6),
    'VR': lambda df: calc_vr(df),
    'VMA5': lambda df: calc_vma(df, 5),
    'DDI': lambda df: calc_ddi(df),
    'VOSC': lambda df: calc_vosc(df),
    'MI12': lambda df: calc_mi(df, 12),
    'DBCD': lambda df: calc_dbcd(df),
    'MTM6': lambda df: calc_mtm(df, 6),
    'MASS': lambda df: calc_mass(df),
    'ROC': lambda df: calc_roc(df),
    'RC': lambda df: calc_rc(df),
    'CR': lambda df: calc_cr(df),
    'TRIX12': lambda df: calc_trix(df, 12),
    'VHF': lambda df: calc_vhf(df),
    'TAPI': lambda df: calc_tapi(df),
    'WAD30': lambda df: calc_wad(df, 30),
}

# ==================== 数据加载 ====================
def load_stock_data(file_path, pool_name):
    print(f"📂 加载 {pool_name}: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['date'])
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    print(f"   股票数量: {df['code'].nunique()}, 记录数量: {len(df)}")
    return df

def load_all_data():
    return {name: load_stock_data(path, name) for name, path in DATA_PATHS.items()}

# ==================== 因子计算 ====================
def calc_by_stock(df, func):
    """按股票分组计算因子，避免数据污染"""
    return df.groupby('code').apply(func).reset_index(drop=True)

# 以下为所有因子函数（每个生成 {FACTOR_NAME}_factor 列）
# 【此处保留完整 calc_xxx 函数定义，与之前相同】

def calc_arbr(df):
    def _single(s):
        s['ar'] = (s['high'] - s['open']).rolling(26).sum() / (s['open'] - s['low']).rolling(26).sum() * 100
        s['br'] = (s['high'] - s['preclose']).rolling(26).sum() / (s['preclose'] - s['low']).rolling(26).sum() * 100
        s['ARBR_factor'] = s['ar'] - s['br']
        return s
    return calc_by_stock(df, _single)

def calc_psy(df, n):
    def _single(s):
        s[f'PSY{n}_factor'] = (s['close'] > s['close'].shift(1)).rolling(n).mean() * 100
        return s
    return calc_by_stock(df, _single)

def calc_pvt(df):
    def _single(s):
        s['PVT_factor'] = ((s['close'] - s['close'].shift(1)) / s['close'].shift(1) * s['volume']).cumsum()
        return s
    return calc_by_stock(df, _single)

def calc_srdm(df, n):
    def _single(s):
        dm = s['high'].diff() - s['low'].diff()
        s[f'SRDM{n}_factor'] = dm.rolling(n).sum()
        return s
    return calc_by_stock(df, _single)

def calc_si(df):
    def _single(s):
        s['SI_factor'] = (s['close'] - s['close'].rolling(20).mean()) / s['close'].rolling(20).std()
        return s
    return calc_by_stock(df, _single)

def calc_micd(df):
    def _single(s):
        s['MICD_factor'] = s['close'].rolling(12).mean() - s['close'].rolling(26).mean()
        return s
    return calc_by_stock(df, _single)

def calc_vroc(df, n):
    def _single(s):
        s[f'VROC{n}_factor'] = (s['volume'] - s['volume'].shift(n)) / s['volume'].shift(n) * 100
        return s
    return calc_by_stock(df, _single)

def calc_boll_mid(df):
    def _single(s):
        s['BOLL_MID_factor'] = s['close'].rolling(20).mean()
        return s
    return calc_by_stock(df, _single)

def calc_obv(df):
    def _single(s):
        s['OBV_factor'] = (np.sign(s['close'].diff()) * s['volume']).fillna(0).cumsum()
        return s
    return calc_by_stock(df, _single)

def calc_wad(df, n):
    def _single(s):
        wad = np.where(s['close'] > s['close'].shift(1), 
                      s['close'] - np.minimum(s['close'].shift(1), s['low']),
                      np.where(s['close'] < s['close'].shift(1),
                              s['close'] - np.maximum(s['close'].shift(1), s['high']), 0))
        s[f'WAD{n}_factor'] = pd.Series(wad).rolling(n).sum()
        return s
    return calc_by_stock(df, _single)

def calc_bbiboll(df):
    def _single(s):
        bbi = (s['close'].rolling(3).mean() + s['close'].rolling(6).mean() + 
               s['close'].rolling(12).mean() + s['close'].rolling(24).mean()) / 4
        upper = bbi + 2 * s['close'].rolling(20).std()
        lower = bbi - 2 * s['close'].rolling(20).std()
        s['BBIBOLL_factor'] = (s['close'] - bbi) / (upper - lower)
        return s
    return calc_by_stock(df, _single)

def calc_bbi(df):
    def _single(s):
        s['BBI_factor'] = (s['close'].rolling(3).mean() + s['close'].rolling(6).mean() + 
                          s['close'].rolling(12).mean() + s['close'].rolling(24).mean()) / 4
        return s
    return calc_by_stock(df, _single)

def calc_mfi(df):
    def _single(s):
        typical = (s['high'] + s['low'] + s['close']) / 3
        mf = typical * s['volume']
        pos_mf = mf.where(typical > typical.shift(1), 0).rolling(14).sum()
        neg_mf = mf.where(typical < typical.shift(1), 0).rolling(14).sum()
        s['MFI_factor'] = 100 - (100 / (1 + pos_mf / neg_mf))
        return s
    return calc_by_stock(df, _single)

def calc_ma(df, n):
    def _single(s):
        s[f'MA{n}_factor'] = s['close'].rolling(n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_priceosc(df):
    def _single(s):
        s['PRICEOSC_factor'] = (s['close'].rolling(12).mean() - s['close'].rolling(26).mean()) / s['close'].rolling(26).mean() * 100
        return s
    return calc_by_stock(df, _single)

def calc_expma(df, n):
    def _single(s):
        s[f'EXPMA{n}_factor'] = s['close'].ewm(span=n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_cdp(df):
    def _single(s):
        cdp = (s['high'].shift(1) + s['low'].shift(1) + s['close'].shift(1)) / 3
        s['CDP_factor'] = s['close'] - cdp
        return s
    return calc_by_stock(df, _single)

def calc_rsi(df):
    def _single(s):
        delta = s['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        s['RSI_factor'] = 100 - (100 / (1 + rs))
        return s
    return calc_by_stock(df, _single)

def calc_atr(df, n):
    def _single(s):
        tr = np.maximum(s['high'] - s['low'], 
                       np.maximum(abs(s['high'] - s['close'].shift(1)),
                                 abs(s['low'] - s['close'].shift(1))))
        s[f'ATR{n}_factor'] = tr.rolling(n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_dpo(df):
    def _single(s):
        ma20 = s['close'].rolling(20).mean()
        s['DPO_factor'] = s['close'] - ma20.shift(10)
        return s
    return calc_by_stock(df, _single)

def calc_rccd(df):
    def _single(s):
        rc = s['close'] / s['close'].shift(20)
        s['RCCD_factor'] = rc - rc.shift(10)
        return s
    return calc_by_stock(df, _single)

def calc_wr(df):
    def _single(s):
        highest = s['high'].rolling(14).max()
        lowest = s['low'].rolling(14).min()
        s['WR_factor'] = (highest - s['close']) / (highest - lowest) * 100
        return s
    return calc_by_stock(df, _single)

def calc_env(df, n):
    def _single(s):
        ma = s['close'].rolling(n).mean()
        s[f'ENV{n}_factor'] = (s['close'] - ma) / ma * 100
        return s
    return calc_by_stock(df, _single)

def calc_vmacd(df):
    def _single(s):
        ema12 = s['volume'].ewm(span=12).mean()
        ema26 = s['volume'].ewm(span=26).mean()
        s['VMACD_factor'] = ema12 - ema26
        return s
    return calc_by_stock(df, _single)

def calc_wr12(df):
    def _single(s):
        highest = s['high'].rolling(12).max()
        lowest = s['low'].rolling(12).min()
        s['WR12_factor'] = (highest - s['close']) / (highest - lowest) * 100
        return s
    return calc_by_stock(df, _single)

def calc_cvlt(df, n):
    def _single(s):
        s[f'CVLT{n}_factor'] = s['close'].rolling(n).std() / s['close'].rolling(n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_cci(df):
    def _single(s):
        tp = (s['high'] + s['low'] + s['close']) / 3
        ma = tp.rolling(20).mean()
        md = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        s['CCI_factor'] = (tp - ma) / (0.015 * md)
        return s
    return calc_by_stock(df, _single)

def calc_volratio(df, n):
    def _single(s):
        s[f'VOLRATIO{n}_factor'] = s['volume'] / s['volume'].rolling(n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_kdj(df):
    def _single(s):
        low9 = s['low'].rolling(9).min()
        high9 = s['high'].rolling(9).max()
        rsv = (s['close'] - low9) / (high9 - low9) * 100
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        s['KDJ_factor'] = k - d
        return s
    return calc_by_stock(df, _single)

def calc_lwr(df):
    def _single(s):
        highest = s['high'].rolling(10).max()
        lowest = s['low'].rolling(10).min()
        s['LWR_factor'] = (highest - s['close']) / (highest - lowest) * 100
        return s
    return calc_by_stock(df, _single)

def calc_dma(df):
    def _single(s):
        s['DMA_factor'] = s['close'].rolling(10).mean() - s['close'].rolling(50).mean()
        return s
    return calc_by_stock(df, _single)

def calc_adtm(df):
    def _single(s):
        dtm = np.where(s['open'] > s['open'].shift(1), 
                      np.maximum(s['high'] - s['open'], s['open'] - s['open'].shift(1)), 0)
        dbm = np.where(s['open'] <= s['open'].shift(1), 
                      np.maximum(s['open'] - s['low'], s['open'].shift(1) - s['open']), 0)
        s['ADTM_factor'] = pd.Series(dtm).rolling(12).sum() - pd.Series(dbm).rolling(12).sum()
        return s
    return calc_by_stock(df, _single)

def calc_srmi(df, n):
    def _single(s):
        s[f'SRMI{n}_factor'] = (s['close'] - s['close'].shift(n)) / s['close'].shift(n)
        return s
    return calc_by_stock(df, _single)

def calc_bias(df):
    def _single(s):
        ma20 = s['close'].rolling(20).mean()
        s['BIAS_factor'] = (s['close'] - ma20) / ma20 * 100
        return s
    return calc_by_stock(df, _single)

def calc_vstd(df, n):
    def _single(s):
        s[f'VSTD{n}_factor'] = s['volume'].rolling(n).std()
        return s
    return calc_by_stock(df, _single)

def calc_macd(df):
    def _single(s):
        ema12 = s['close'].ewm(span=12).mean()
        ema26 = s['close'].ewm(span=26).mean()
        s['MACD_factor'] = ema12 - ema26
        return s
    return calc_by_stock(df, _single)

def calc_vrsi(df, n):
    def _single(s):
        delta = s['volume'].diff()
        gain = delta.where(delta > 0, 0).rolling(n).mean()
        loss = (-delta).where(delta < 0, 0).rolling(n).mean()
        rs = gain / loss
        s[f'VRSI{n}_factor'] = 100 - (100 / (1 + rs))
        return s
    return calc_by_stock(df, _single)

def calc_vr(df):
    def _single(s):
        av = s['volume'].where(s['close'] > s['close'].shift(1), 0).rolling(26).sum()
        bv = s['volume'].where(s['close'] < s['close'].shift(1), 0).rolling(26).sum()
        s['VR_factor'] = av / bv
        return s
    return calc_by_stock(df, _single)

def calc_vma(df, n):
    def _single(s):
        s[f'VMA{n}_factor'] = s['volume'].rolling(n).mean()
        return s
    return calc_by_stock(df, _single)

def calc_ddi(df):
    def _single(s):
        tr = np.maximum(s['high'] - s['low'], 
                       np.maximum(abs(s['high'] - s['close'].shift(1)),
                                 abs(s['low'] - s['close'].shift(1))))
        s['DDI_factor'] = tr.rolling(14).mean()
        return s
    return calc_by_stock(df, _single)

def calc_vosc(df):
    def _single(s):
        s['VOSC_factor'] = s['volume'].rolling(12).mean() - s['volume'].rolling(26).mean()
        return s
    return calc_by_stock(df, _single)

def calc_mi(df, n):
    def _single(s):
        s[f'MI{n}_factor'] = s['close'].diff(n)
        return s
    return calc_by_stock(df, _single)

def calc_dbcd(df):
    def _single(s):
        s['DBCD_factor'] = (s['close'].diff(10) - s['close'].diff(20))
        return s
    return calc_by_stock(df, _single)

def calc_mtm(df, n):
    def _single(s):
        s[f'MTM{n}_factor'] = s['close'] - s['close'].shift(n)
        return s
    return calc_by_stock(df, _single)

def calc_mass(df):
    def _single(s):
        hl = s['high'] - s['low']
        s['MASS_factor'] = hl.rolling(9).sum() / hl.rolling(25).sum()
        return s
    return calc_by_stock(df, _single)

def calc_roc(df):
    def _single(s):
        s['ROC_factor'] = (s['close'] - s['close'].shift(10)) / s['close'].shift(10) * 100
        return s
    return calc_by_stock(df, _single)

def calc_rc(df):
    def _single(s):
        s['RC_factor'] = s['close'] / s['close'].shift(20)
        return s
    return calc_by_stock(df, _single)

def calc_cr(df):
    def _single(s):
        s['CR_factor'] = (s['volume'] * s['close'].diff(5)).rolling(10).sum()
        return s
    return calc_by_stock(df, _single)

def calc_trix(df, n):
    def _single(s):
        s[f'TRIX{n}_factor'] = s['close'].ewm(span=n).mean().pct_change()
        return s
    return calc_by_stock(df, _single)

def calc_vhf(df):
    def _single(s):
        s['VHF_factor'] = (s['close'].rolling(28).max() - s['close'].rolling(28).min()) / s['close'].rolling(28).std()
        return s
    return calc_by_stock(df, _single)

def calc_tapi(df):
    def _single(s):
        s['TAPI_factor'] = s['close'] * s['volume'] / s['close'].rolling(10).mean()
        return s
    return calc_by_stock(df, _single)

# ==================== 回测与绩效 ====================
class FactorBacktest:
    def __init__(self, df, factor_name, n_groups=5):
        self.df = df.copy()
        self.factor_name = factor_name
        self.col_name = factor_name + '_factor'
        self.n_groups = n_groups

    def run_backtest(self):
        clean = self.df.dropna(subset=[self.col_name, 'pctChg']).copy()
        if clean.empty:
            print("   ⚠️  清洗后无有效数据，跳过回测")
            return self
            
        daily_g, daily_ic, skipped, processed = [], [], 0, 0
        
        for dt, g in clean.groupby('date'):
            if len(g) < self.n_groups * 5:
                skipped += 1
                continue
                
            processed += 1
            
            # 检查唯一值数量
            unique_vals = g[self.col_name].nunique()
            if unique_vals < self.n_groups:
                # 降级方案1：rank分桶
                g['group'] = (g[self.col_name].rank() / (len(g) / self.n_groups)).astype(int).clip(0, self.n_groups-1)
            else:
                try:
                    # 首选：qcut
                    g['group'] = pd.qcut(g[self.col_name], self.n_groups, labels=range(self.n_groups), duplicates='drop')
                except:
                    # 降级方案2：cut
                    try:
                        g['group'] = pd.cut(g[self.col_name], bins=self.n_groups, labels=range(self.n_groups), include_lowest=True)
                    except:
                        # 降级方案3：rank
                        g['group'] = (g[self.col_name].rank() / (len(g) / self.n_groups)).astype(int).clip(0, self.n_groups-1)
            
            # 计算收益
            ret = g.groupby('group')['pctChg'].mean()
            ls = ret.iloc[-1] - ret.iloc[0] if len(ret) == self.n_groups else np.nan
            ic = g[self.col_name].corr(g['pctChg'], method='spearman')
            
            daily_g.append({'date': dt, **{f'group_{i}': ret.iloc[i] if i < len(ret) else np.nan for i in range(self.n_groups)}})
            daily_ic.append({'date': dt, 'ic': ic})
        
        print(f"   处理完成：总日期数 {len(clean['date'].unique())}，有效 {processed} 天，跳过 {skipped} 天")
        
        # 修复：检查是否有有效数据
        if not daily_g:
            print("   ⚠️  无有效分组数据，回测结果为空")
            return self
            
        self.group_returns = pd.DataFrame(daily_g).set_index('date')
        self.ic_series = pd.DataFrame(daily_ic).set_index('date')['ic']
        self.cumulative_returns = (1 + self.group_returns / 100).cumprod()
        self.long_short_returns = self.group_returns[f'group_{self.n_groups-1}'] - self.group_returns['group_0']
        self.long_short_cumulative = (1 + self.long_short_returns / 100).cumprod()
        return self

    def calc_turnover(self):
        return pd.DataFrame([{'group': f'Group_{i+1}', 'turnover': self.group_returns[f'group_{i}'].diff().abs().mean()} for i in range(self.n_groups)])

def perf_metrics(ret, name):
    if ret.isna().all() or len(ret) == 0:
        return pd.Series({'策略名称': name, '年化收益率': 'N/A', '年化波动率': 'N/A', '夏普比率': 'N/A', '最大回撤': 'N/A', '日胜率': 'N/A', '总交易日': 0})
    ann_ret = (1 + ret / 100).prod() ** (252 / len(ret)) - 1
    vol = ret.std() * np.sqrt(252 / 100)
    sharpe = ann_ret / vol if vol else 0
    cum = (1 + ret / 100).cumprod()
    dd = (cum - cum.expanding().max()) / cum.expanding().max()
    win = (ret > 0).mean()
    return pd.Series({'策略名称': name, '年化收益率': f"{ann_ret:.2%}", '年化波动率': f"{vol:.2%}", '夏普比率': f"{sharpe:.2f}", '最大回撤': f"{dd.min():.2%}", '日胜率': f"{win:.2%}", '总交易日': len(ret)})

# ==================== 可视化 ====================
class Visualizer:
    def __init__(self, bt, pool, out_dir):
        self.bt, self.pool, self.out_dir = bt, pool, Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, name):
        fig.savefig(self.out_dir / f"{self.pool}_{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_all(self):
        self.plot_cum()
        self.plot_perf()
        self.plot_ls()
        self.plot_turn()
        self.plot_ic()

    def plot_cum(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, self.bt.n_groups))
        for i in range(self.bt.n_groups):
            ax.plot(self.bt.cumulative_returns.index, (self.bt.cumulative_returns[f'group_{i}'] - 1) * 100,
                    label=f'Group {i+1}', color=colors[i], linewidth=1.5)
        if hasattr(self.bt, 'long_short_cumulative'):
            ax.plot(self.bt.long_short_cumulative.index, (self.bt.long_short_cumulative - 1) * 100,
                    label='Long-Short', color='purple', linewidth=2, linestyle='--')
        ax.set_title(f'{self.pool} {self.bt.factor_name} 分组收益曲线', fontsize=16)
        ax.legend(); ax.grid(alpha=0.3)
        self.save(fig, 'cumulative_returns')

    def plot_perf(self):
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('off')
        perf_data = [perf_metrics(self.bt.group_returns[f'group_{i}'], f'Group {i+1}') for i in range(self.bt.n_groups)]
        if hasattr(self.bt, 'long_short_returns'):
            perf_data.append(perf_metrics(self.bt.long_short_returns, 'Long-Short'))
        df = pd.DataFrame(perf_data)
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.5)
        ax.set_title(f'{self.pool} {self.bt.factor_name} 分组绩效分析', fontsize=16, pad=20)
        self.save(fig, 'performance_analysis')

    def plot_ls(self):
        if not hasattr(self.bt, 'long_short_returns') or self.bt.long_short_returns.isna().all(): return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(self.bt.long_short_cumulative.index, (self.bt.long_short_cumulative - 1) * 100, color='purple', lw=2)
        ax1.set_title('多空组合累计收益'); ax1.grid(alpha=0.3)
        monthly = self.bt.long_short_returns.resample('M').sum()
        ax2.bar(range(len(monthly)), monthly.values, color=['red' if x > 0 else 'green' for x in monthly], alpha=0.7)
        ax2.set_title('多空组合月度收益分布'); ax2.grid(alpha=0.3)
        self.save(fig, 'long_short_analysis')

    def plot_turn(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        turnover_df = self.bt.calc_turnover()
        bars = ax.bar(turnover_df['group'], turnover_df['turnover'], color='skyblue', alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        ax.set_title(f'{self.pool} {self.bt.factor_name} 分组换手率分析'); ax.grid(axis='y', alpha=0.3)
        self.save(fig, 'turnover_analysis')

    def plot_ic(self):
        if self.bt.ic_series.isna().all(): return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(self.bt.ic_series.index, self.bt.ic_series.values, color='blue', lw=1.5, alpha=0.7)
        ax1.axhline(y=0, color='black', ls='--', alpha=0.5)
        ax1.axhline(y=self.bt.ic_series.mean(), color='red', ls='-', label=f'均值: {self.bt.ic_series.mean():.3f}')
        ax1.set_title('信息系数(IC)时间序列'); ax1.legend(); ax1.grid(alpha=0.3)
        ax2.hist(self.bt.ic_series.dropna().values, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        ax2.set_title('IC值分布'); ax2.grid(alpha=0.3)
        self.save(fig, 'ic_analysis')

# ==================== 文字小结 ====================
def write_text_summary(pool, bt, factor, out_dir, sd, ed):
    lines = [f"# {pool} {factor} 因子回测报告", "="*50, ""]
    lines += [f"- 回测周期: {sd} 至 {ed}",
              f"- 因子名称: {factor}",
              f"- 分组数量: {N_GROUPS}",
              f"- 回测天数: {len(bt.group_returns) if hasattr(bt, 'group_returns') else 0}",
              f"- IC均值: {bt.ic_series.mean() if hasattr(bt, 'ic_series') else 0:.4f}",
              f"- IC标准差: {bt.ic_series.std() if hasattr(bt, 'ic_series') else 0:.4f}", ""]
    
    if hasattr(bt, 'group_returns') and len(bt.group_returns) > 0:
        perf_data = [perf_metrics(bt.group_returns[f'group_{i}'], f'Group {i+1}') for i in range(bt.n_groups)]
        if hasattr(bt, 'long_short_returns'):
            perf_data.append(perf_metrics(bt.long_short_returns, 'Long-Short'))
        df = pd.DataFrame(perf_data)
        lines += ["## 分组绩效", "```", df.to_string(index=False), "```", ""]
        
        turnover_df = bt.calc_turnover()
        lines += ["## 换手率"]
        for _, r in turnover_df.iterrows():
            lines.append(f"- {r['group']}: {r['turnover']:.2f}%")
    else:
        lines += ["## 回测结果", "⚠️ 无有效回测数据"]
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{pool}_summary_report.txt").write_text('\n'.join(lines), encoding='utf-8')

# ==================== 主流程 ====================
def run_factor(factor_name, factor_func, sd, ed):
    print(f"\n{'='*60}")
    print(f">>> 开始因子 {factor_name}")
    print(f"{'='*60}")
    out_dir = RES_ROOT / factor_name
    data_dict = load_all_data()
    for pool, df in data_dict.items():
        print(f"\n>>> 处理 {pool} ...")
        df_with_factor = factor_func(df)
        bt = FactorBacktest(df_with_factor, factor_name).run_backtest()
        if hasattr(bt, 'group_returns') and len(bt.group_returns) > 0:
            Visualizer(bt, pool, out_dir).plot_all()
            write_text_summary(pool, bt, factor_name, out_dir, sd, ed)
        else:
            print(f"   ⚠️  {pool} 无有效回测数据")
    print(f"✅ {factor_name} 完成！")

def main():
    print("="*60)
    print("通用多因子回测框架（最终修复版）")
    print(f"待测因子数: {len(FACTORS)}")
    print("="*60)
    
    # 可以指定只跑部分因子（用于调试）
    # target_factors = ['ARBR', 'PSY12', 'MACD']
    target_factors = list(FACTORS.keys())
    
    for name in target_factors:
        run_factor(name, FACTORS[name],START_DATE, END_DATE)
    
    print("\n" + "="*60)
    print("✅ 所有因子完成！")
    print(f"结果保存在: {RES_ROOT}")
    print("="*60)

if __name__ == "__main__":
    main()
