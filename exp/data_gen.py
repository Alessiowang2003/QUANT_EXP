import baostock as bs
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# =============================================================================
# 配置参数 自己填回测时间
# =============================================================================
START_DATE = ""
END_DATE = ""

# =============================================================================
# 底下不要改
# =============================================================================
# 路径配置：脚本在exp文件夹，输出到同级的data文件夹
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
OUTPUT_FILES = {
    'hs300': os.path.join(DATA_DIR, "hs300_stocks_2024.csv"),
    'zz500': os.path.join(DATA_DIR, "zz500_stocks_2024.csv"),
    'zz800': os.path.join(DATA_DIR, "zz800_stocks_2024.csv")
}

# =============================================================================
# 核心函数
# =============================================================================

def login_baostock():
    """登录baostock"""
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return False
    print("✅ baostock登录成功")
    return True

def get_index_stocks(index_type, start_date):
    """
    获取指数成分股列表
    :param index_type: 'hs300' 或 'zz500'
    :param start_date: 查询日期
    """
    if index_type == 'hs300':
        rs = bs.query_hs300_stocks(date=start_date)
    elif index_type == 'zz500':
        rs = bs.query_zz500_stocks(date=start_date)
    else:
        print(f"❌ 不支持的指数类型: {index_type}")
        return []
    
    if rs.error_code != '0':
        print(f"❌ 获取{index_type}成分股失败: {rs.error_msg}")
        return []
    
    stocks = []
    while rs.next():
        stocks.append(rs.get_row_data())
    
    df = pd.DataFrame(stocks, columns=rs.fields)
    stock_list = df['code'].tolist()
    print(f"📊 获取到 {len(stock_list)} 只{index_type}成分股")
    return stock_list

def get_stock_data(code, start_date, end_date):
    """获取单只股票数据（含重试机制）"""
    fields = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
    
    for attempt in range(3):
        try:
            rs = bs.query_history_k_data_plus(
                code=code,
                fields=fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            
            if rs.error_code != '0':
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            data = []
            while rs.next():
                data.append(rs.get_row_data())
            
            if not data:
                return None
            
            df = pd.DataFrame(data, columns=rs.fields)
            
            # 数据类型转换
            df['date'] = pd.to_datetime(df['date'])
            numeric_cols = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'pctChg']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 清洗数据
            df = df.dropna(subset=['open', 'close', 'volume'])
            df = df[df['volume'] > 0]
            
            return df
            
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None

def fetch_index_data(index_type, start_date, end_date, output_file):
    """
    获取指数成分股数据并保存
    :param index_type: 'hs300' 或 'zz500'
    """
    print("="*60)
    print(f"{index_type.upper()}成分股数据获取程序")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"输出文件: {output_file}")
    print("="*60)
    
    # 获取股票列表
    stock_codes = get_index_stocks(index_type, start_date)
    if not stock_codes:
        return False
    
    all_data = []
    success_count = 0
    
    # 遍历获取数据
    for idx, code in enumerate(stock_codes, 1):
        print(f"\n[{idx:03d}/{len(stock_codes)}] {code}")
        
        df = get_stock_data(code, start_date, end_date)
        
        if df is not None and not df.empty:
            all_data.append(df)
            success_count += 1
            print(f"  ✅ 成功: {len(df)} 条记录")
        else:
            print(f"  ❌ 失败")
        
        # 每50只暂停
        if idx % 50 == 0:
            print("\n⏸️  暂停5秒...")
            time.sleep(5)
    
    # 合并并保存
    if all_data:
        print("\n" + "="*60)
        print("正在合并数据...")
        
        df_all = pd.concat(all_data, ignore_index=True)
        df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"✅ 合并完成！总记录数: {len(df_all)}")
        print(f"📁 文件已保存: {output_file}")
        print(f"\n股票数量: {success_count}/{len(stock_codes)}")
        
        return True
    else:
        print("❌ 未获取到任何数据")
        return False

def generate_zz800():
    """合并沪深300和中证500数据生成中证800（自动去重）"""
    print("\n" + "="*60)
    print("中证800数据生成程序 (合并沪深300 + 中证500)")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(OUTPUT_FILES['hs300']):
        print(f"❌ 文件不存在: {OUTPUT_FILES['hs300']}")
        return False
    if not os.path.exists(OUTPUT_FILES['zz500']):
        print(f"❌ 文件不存在: {OUTPUT_FILES['zz500']}")
        return False
    
    print(f"📂 正在读取沪深300数据...")
    df_hs300 = pd.read_csv(OUTPUT_FILES['hs300'], parse_dates=['date'])
    
    print(f"📂 正在读取中证500数据...")
    df_zz500 = pd.read_csv(OUTPUT_FILES['zz500'], parse_dates=['date'])
    
    # 统计原始数据
    hs300_codes = set(df_hs300['code'].unique())
    zz500_codes = set(df_zz500['code'].unique())
    
    print(f"\n📊 数据概览:")
    print(f"  沪深300: {len(hs300_codes)} 只股票, {len(df_hs300)} 条记录")
    print(f"  中证500: {len(zz500_codes)} 只股票, {len(df_zz500)} 条记录")
    
    # 合并数据
    print("\n🔀 正在合并数据...")
    df_combined = pd.concat([df_hs300, df_zz500], ignore_index=True)
    
    # 去重（保留第一条出现的数据）
    df_zz800 = df_combined.drop_duplicates(subset=['code', 'date'], keep='first')
    
    # 排序
    df_zz800 = df_zz800.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 统计合并后数据
    zz800_codes = set(df_zz800['code'].unique())
    overlap_codes = hs300_codes & zz500_codes
    
    print(f"\n📊 合并结果:")
    print(f"  重叠股票: {len(overlap_codes)} 只")
    print(f"  中证800: {len(zz800_codes)} 只股票, {len(df_zz800)} 条记录")
    
    # 保存文件
    print(f"\n💾 正在保存至 {OUTPUT_FILES['zz800']}...")
    df_zz800.to_csv(OUTPUT_FILES['zz800'], index=False, encoding='utf-8-sig')
    
    print("="*60)
    print("✅ 中证800生成完成！")
    print(f"📁 文件已保存: {OUTPUT_FILES['zz800']}")
    print("="*60)
    
    # 显示前5行
    print("\n数据样例:")
    print(df_zz800.head())
    
    return True

def main():
    """主执行函数"""
    print("沪深300/中证500/中证800数据获取程序")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确保data目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 登录baostock
    if not login_baostock():
        return
    
    try:
        # 1. 获取沪深300数据
        if not fetch_index_data('hs300', START_DATE, END_DATE, OUTPUT_FILES['hs300']):
            print("❌ 沪深300数据获取失败，程序终止")
            return
        
        # 2. 获取中证500数据
        if not fetch_index_data('zz500', START_DATE, END_DATE, OUTPUT_FILES['zz500']):
            print("❌ 中证500数据获取失败，程序终止")
            return
        
        # 3. 登出baostock
        bs.logout()
        
        # 4. 生成中证800数据
        generate_zz800()
        
        print("\n🎉 所有任务完成！")
        
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        bs.logout()

if __name__ == "__main__":
    main()