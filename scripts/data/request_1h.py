import csv
import datetime
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

# --- 1. 参数设置 ---
SYMBOL = 'BTCUSDT'
INTERVAL = '1h'  # 1小时数据

# 建议使用 UTC 时间进行请求，避免本地时区干扰
# 资金费率从 2019-09-10 08:00:00 UTC 开始有数据
REQUEST_START_DATE = datetime.datetime(2019, 9, 10, 8, 0, 0, tzinfo=datetime.timezone.utc)
REQUEST_END_DATE = datetime.datetime(2025, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)

OUTPUT_FILE = f'data/raw/{SYMBOL}_{INTERVAL}_binance_data.csv'
BASE_SPOT_URL = "https://api.binance.com"
BASE_FUTURES_URL = "https://fapi.binance.com"
HOUR_MS = 60 * 60 * 1000

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "nusri-project/1.0"})

# --- 2. 辅助函数：获取带重试机制的数据 ---
def fetch_json(url: str, params: Dict[str, object], max_retries: int = 5) -> List:
    for i in range(max_retries):
        try:
            response = SESSION.get(url, params=params, timeout=20)  # 增加超时设置
            if response.status_code >= 400:
                detail = (response.text or "").strip().replace("\n", " ")
                if len(detail) > 300:
                    detail = detail[:300] + "..."
                raise requests.exceptions.HTTPError(
                    f"{response.status_code} Client Error; detail={detail}",
                    response=response,
                )
            return response.json()
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # 400 on these futures endpoints is common when the requested window is invalid/unsupported;
            # don't waste time retrying identical requests many times.
            retries = 1 if status == 400 else max_retries
            print(f"  [警告] 请求失败 (尝试 {i+1}/{retries}): {e}")
            if status == 400 or i + 1 >= retries:
                return []
            time.sleep(2 * (i + 1))
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  [警告] 请求失败 (尝试 {i+1}/{max_retries}): {e}")
            time.sleep(2 * (i + 1))  # 失败后指数退避等待
    
    print("  [错误] 多次重试失败，跳过该段数据。")
    return []


def fetch_klines(symbol: str, interval: str, start_ts: int, end_ts: int, limit: int = 1000) -> List:
    url = f"{BASE_SPOT_URL}/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': limit
    }
    return fetch_json(url, params)


def fetch_time_series(
    url: str,
    base_params: Dict[str, object],
    start_ts: int,
    end_ts: int,
    *,
    time_key: str,
    step_ms: int,
    limit: int,
    max_window_ms: Optional[int] = None,
) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    current_start = start_ts

    while current_start < end_ts:
        window_end = end_ts if max_window_ms is None else min(end_ts, current_start + max_window_ms)
        params = dict(base_params)
        params.update({
            "startTime": current_start,
            "endTime": window_end,
            "limit": limit,
        })
        rows = fetch_json(url, params)
        if not rows:
            # Some endpoints return 4xx when requesting time ranges with no available data.
            # To avoid aborting the whole job, advance in window-sized steps when configured.
            if max_window_ms is None:
                break
            current_start = window_end + step_ms
            continue

        all_rows.extend(rows)
        last_ts = int(rows[-1][time_key])
        next_start = last_ts + step_ms
        if next_start <= current_start:
            break
        current_start = next_start
        time.sleep(0.1)

    return all_rows


def build_funding_curve(rows: Iterable[Dict[str, object]]) -> List[Tuple[int, str]]:
    curve = []
    for row in rows:
        if "fundingTime" in row and "fundingRate" in row:
            curve.append((int(row["fundingTime"]), str(row["fundingRate"])))
    return sorted(curve, key=lambda item: item[0])

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

# --- 3.5 获取资金费率 ---
print("\n=== 开始获取衍生数据 (期货接口) ===")
now_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
futures_metrics_end_ms = min(end_timestamp_ms, now_ms)
funding_rows = fetch_time_series(
    f"{BASE_FUTURES_URL}/fapi/v1/fundingRate",
    {"symbol": SYMBOL},
    start_timestamp_ms,
    futures_metrics_end_ms,
    time_key="fundingTime",
    step_ms=HOUR_MS * 8,
    limit=1000,
)
funding_curve = build_funding_curve(funding_rows)
print(f"Funding Rate 数据条数: {len(funding_curve)}")

# --- 4. 写入 CSV ---
if all_klines:
    print(f"\n数据获取完毕，共 {len(all_klines)} 条。正在写入 {OUTPUT_FILE} ...")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as file:
        # 使用分号 ; 作为分隔符（符合你之前的格式）
        writer = csv.writer(file, delimiter=';')
        
        # 修改表头：
        # 1. 保留基础 OHLCV + 成交额字段
        # 2. 增加主动买入量、持仓量、资金费率
        header = [
            'date',
            'open',
            'high',
            'low',
            'close',
            'volume',
            'amount',
            'taker_buy_base_volume',
            'taker_buy_quote_volume',
            'funding_rate',
        ]
        writer.writerow(header)

        funding_idx = 0
        funding_rate: Optional[str] = None

        for kline in all_klines:
            # Binance API 返回结构:
            # [0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume (BTC), ..., 7: Quote Volume (USDT),
            #  9: Taker buy base asset volume, 10: Taker buy quote asset volume]
            
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
            taker_buy_base = kline[9]
            taker_buy_quote = kline[10]

            while funding_idx < len(funding_curve) and funding_curve[funding_idx][0] <= open_time_ms:
                funding_rate = funding_curve[funding_idx][1]
                funding_idx += 1

            writer.writerow([
                date_str,
                open_p,
                high_p,
                low_p,
                close_p,
                vol_btc, 
                vol_usdt,
                taker_buy_base,
                taker_buy_quote,
                funding_rate or "",
            ])
            
    print("写入完成！")
else:
    print("未获取到数据。")
