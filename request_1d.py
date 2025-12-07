import requests
import datetime
import csv
import time

# --- 1. 参数设置 ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1d'     # 【修改点】这里改为 1d 代表日线数据

# 币安 BTCUSDT 最早从 2017-08-17 开始
# 使用 UTC 时区
REQUEST_START_DATE = datetime.datetime(2017, 8, 17, tzinfo=datetime.timezone.utc)
# 结束时间设置得比较远，确保能获取到运行当天的最新数据
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
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  [警告] 请求失败 (尝试 {i+1}/{max_retries}): {e}")
            time.sleep(2 * (i + 1))
    
    print("  [错误] 多次重试失败，跳过该段数据。")
    return []

# --- 3. 主逻辑 ---

start_timestamp_ms = int(REQUEST_START_DATE.timestamp() * 1000)
end_timestamp_ms = int(REQUEST_END_DATE.timestamp() * 1000)

all_klines = []
current_start_time = start_timestamp_ms

print(f"=== 开始获取 {SYMBOL} {INTERVAL} (日线) 数据 ===")
print(f"时间范围 (UTC): {REQUEST_START_DATE} 到 {REQUEST_END_DATE}")

while current_start_time < end_timestamp_ms:
    klines = fetch_klines(SYMBOL, INTERVAL, current_start_time, end_timestamp_ms)
    
    if not klines:
        break
    
    all_klines.extend(klines)
    
    last_timestamp = klines[-1][0]
    current_start_time = last_timestamp + 1
    
    # 打印进度
    last_date_obj = datetime.datetime.fromtimestamp(last_timestamp / 1000, datetime.timezone.utc)
    print(f"已收集: {len(all_klines)} 天 | 最新日期: {last_date_obj.strftime('%Y-%m-%d')}")
    
    time.sleep(0.1)

# --- 4. 写入 CSV ---
if all_klines:
    print(f"\n数据获取完毕，共 {len(all_klines)} 条。正在写入 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        
        # 表头保持一致
        header = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        writer.writerow(header)

        for kline in all_klines:
            open_time_ms = kline[0]
            
            # 即使是日线，也建议保留 %H:%M:%S (通常是 00:00:00)，这样 Qlib 识别最稳定
            dt_obj = datetime.datetime.fromtimestamp(open_time_ms / 1000, datetime.timezone.utc)
            date_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            
            open_p = kline[1]
            high_p = kline[2]
            low_p = kline[3]
            close_p = kline[4]
            vol_btc = kline[5]
            vol_usdt = kline[7]

            writer.writerow([
                date_str,
                open_p,
                high_p,
                low_p,
                close_p,
                vol_btc, 
                vol_usdt
            ])
            
    print(f"成功！文件已保存为: {OUTPUT_FILE}")
else:
    print("未获取到数据。")