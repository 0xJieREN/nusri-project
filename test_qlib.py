import qlib
from qlib.data import D

# 1. 初始化：指向你刚才生成的 binary 数据目录
# provider_uri 必须是绝对路径或相对于当前脚本的路径
qlib.init(provider_uri='./qlib_data/my_crypto_data')
    
# 2. 测试：获取 BTC 数据
# market='all' 会去读取 instruments/all.txt，dump_bin 自动生成了这个文件
df = D.features(instruments=['BTCUSDT'], 
                fields=['$close', '$open', '$high', '$low', '$volume', '$amount', '$vwap'],
                start_time='2020-01-01 00:00:00',
                end_time='2020-12-31 23:00:00',
                freq='60min',)

print(df)
