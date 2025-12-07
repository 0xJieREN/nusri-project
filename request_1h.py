import requests
import datetime
import csv
import time

# --- 1. 参数设置 ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'  # 1小时数据

# 建议使用 UTC 时间进行请求，避免本地时区干扰
# 这里设置为从 2017-08-17 00:00:00 UTC 开始
REQUEST_START_DATE = datetime.datetime(2017, 8, 17, tzinfo=datetime.timezone.utc)
REQUEST_END_DATE = datetime.datetime(2025, 12, 1, tzinfo=datetime.timezone.utc)

OUTPUT_FILE = f'{SYMBOL}_{INTERVAL}_binance_data.csv'

# --- 2. 辅助函数：获取带重试机制的数据 ---
def fetch_klines(symbol, interval, start_ts, end_ts, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': limit
    }
    
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10) # 增加超时设置
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  [警告] 请求失败 (尝试 {i+1}/{max_retries}): {e}")
            time.sleep(2 * (i + 1)) # 失败后指数退避等待
    
    print("  [错误] 多次重试失败，跳过该段数据。")
    return []

# --- 3. 主逻辑 ---

# 转换为毫秒时间戳
start_timestamp_ms = int(REQUEST_START_DATE.timestamp() * 1000)
end_timestamp_ms = int(REQUEST_END_DATE.timestamp() * 1000)

all_klines = []
current_start_time = start_timestamp_ms

print(f"=== 开始获取 {SYMBOL} {INTERVAL} 数据 ===")
print(f"时间范围 (UTC): {REQUEST_START_DATE} 到 {REQUEST_END_DATE}")

while current_start_time < end_timestamp_ms:
    # 获取数据
    klines = fetch_klines(SYMBOL, INTERVAL, current_start_time, end_timestamp_ms)
    
    if not klines:
        break
    
    all_klines.extend(klines)
    
    # 更新下一次请求的开始时间
    last_timestamp = klines[-1][0]
    current_start_time = last_timestamp + 1
    
    # 打印进度
    # 转换最后一条数据的时间用于显示 (强制使用 UTC)
    last_date_obj = datetime.datetime.fromtimestamp(last_timestamp / 1000, datetime.timezone.utc)
    print(f"已收集: {len(all_klines)} 条 | 最新时间: {last_date_obj.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 稍微等待，避免触发频率限制
    time.sleep(0.1)

# --- 4. 写入 CSV ---
if all_klines:
    print(f"\n数据获取完毕，共 {len(all_klines)} 条。正在写入 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        # 使用分号 ; 作为分隔符（符合你之前的格式）
        writer = csv.writer(file, delimiter=';')
        
        # 修改表头：
        # 1. 增加了 'volume_btc' (基础成交量) 和 'volume_usdt' (成交额)
        # 2. 区分清楚，方便后续因子计算
        header = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        writer.writerow(header)

        for kline in all_klines:
            # Binance API 返回结构:
            # [0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume (BTC), ..., 7: Quote Volume (USDT)]
            
            open_time_ms = kline[0]
            
            # 【核心修改】：
            # 1. 使用 datetime.timezone.utc 确保是 UTC 时间
            # 2. 格式化字符串增加 %H:%M:%S，精确到秒
            dt_obj = datetime.datetime.fromtimestamp(open_time_ms / 1000, datetime.timezone.utc)
            date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            
            open_p = kline[1]
            high_p = kline[2]
            low_p = kline[3]
            close_p = kline[4]
            vol_btc = kline[5]  # 成交量 (币) -> 对应 Qlib 的 volume
            vol_usdt = kline[7] # 成交额 (钱) -> 对应 Qlib 的 amount/turnover

            writer.writerow([
                date_str,
                open_p,
                high_p,
                low_p,
                close_p,
                vol_btc, 
                vol_usdt
            ])
            
    print("写入完成！")
else:
    print("未获取到数据。")