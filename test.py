import csv
import os

# --- 1. 参数设置 ---
# 必须与上一个脚本生成的文件名一致
INPUT_FILE = 'BTCUSDT_1d_binance_data.csv' 
TOP_N = 100  # 你想打印排名前多少的数据

# --- 2. 读取并处理数据 ---
if not os.path.exists(INPUT_FILE):
    print(f"错误: 找不到文件 '{INPUT_FILE}'。请先运行上一个脚本获取数据。")
else:
    print(f"正在读取文件 '{INPUT_FILE}' ...")
    
    rows_with_amount = []
    
    try:
        with open(INPUT_FILE, mode='r', encoding='utf-8') as f:
            # 注意：上一个脚本使用的是分号 ; 作为分隔符
            reader = csv.reader(f, delimiter=';')
            
            # 读取表头
            header = next(reader)
            
            # 找到 'amount' 和 'close' 在哪一列 (防止列顺序变动)
            try:
                date_idx = header.index('date')
                close_idx = header.index('close')
                amount_idx = header.index('amount')
            except ValueError:
                # 如果找不到表头，默认使用上个脚本的索引位置
                date_idx = 0
                close_idx = 4
                amount_idx = 6

            # 遍历每一行数据
            for row in reader:
                try:
                    # 提取需要的数据，并将数字字符串转换为 float 以便排序
                    item = {
                        'date': row[date_idx],
                        'close': row[close_idx],
                        'amount': float(row[amount_idx]) # 转换为浮点数
                    }
                    rows_with_amount.append(item)
                except (ValueError, IndexError):
                    # 跳过可能存在的坏数据行
                    continue

        # --- 3. 排序 ---
        # key参数指定按 'amount' 排序，reverse=True 表示降序（从大到小）
        print(f"读取完毕，共 {len(rows_with_amount)} 条数据。正在按交易额排序...")
        rows_with_amount.sort(key=lambda x: x['amount'], reverse=True)

        # --- 4. 打印输出 ---
        print("\n" + "="*70)
        print(f"BTCUSDT 天级别 - 历史交易额排名 TOP {TOP_N}")
        print("="*70)
        # 打印表头，使用格式化字符串对齐
        print(f"{'排名':<5} {'日期时间 (UTC)':<25} {'交易额 (USDT)':<20} {'收盘价'}")
        print("-" * 70)

        for i, item in enumerate(rows_with_amount[:TOP_N]):
            rank = i + 1
            date = item['date']
            # {:,.2f} 表示添加千位分隔符并保留两位小数
            amount = f"{item['amount']:,.2f}" 
            close = item['close']
            
            print(f"{rank:<5} {date:<25} {amount:<20} {close}")
            
        print("="*70)

    except Exception as e:
        print(f"处理过程中发生错误: {e}")