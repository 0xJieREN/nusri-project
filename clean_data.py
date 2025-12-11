import pandas as pd
import os

# 1. 读取你之前下载的数据
df = pd.read_csv('BTCUSDT_1d_binance_data.csv', sep=';')

# 2. 增加 symbol 列 (Qlib 必须项)
df['symbol'] = 'BTCUSDT'

# 3. 确保日期格式正确 (去掉时分秒，只留日期)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

# 4. 加密货币没有复权，所以因子永远是 1.0
df['factor'] = 1.0 

# 如果有 'amount' 列，可以保留作为额外特征，但基础 OHLCV 必须有

# 5. 保存到一个新的文件夹，每个币种一个 CSV (虽然你只有一个)
output_dir = 'qlib_source_data'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f'{output_dir}/BTCUSDT.csv', index=False)

print("Qlib 源数据准备完成！")