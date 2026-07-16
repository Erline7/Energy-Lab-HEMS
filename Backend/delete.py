import sqlite3
import os
import sys

# 强制将标准输出设置为 UTF-8，支持打印 Emoji
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# 你的其他代码...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hems_data.db").replace("\\", "/")

try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    delete_date = '2020-01-01'

    # 彻底清除 5 张表中的测试数据
    cursor.execute(f"DELETE FROM telemetry_main WHERE timestamp >= '{delete_date}'")
    cursor.execute(f"DELETE FROM water_main_logs WHERE timestamp >= '{delete_date}'")
    cursor.execute(f"DELETE FROM water_sub_logs WHERE timestamp >= '{delete_date}'")
    cursor.execute(f"DELETE FROM natural_gas_logs WHERE month_date >= '{delete_date}'")

    # 👇 就是这一行！上次漏掉的 NILM 拆解表清理
    cursor.execute(f"DELETE FROM nilm_inferred_logs WHERE timestamp >= '{delete_date}'")

    conn.commit()
    print("✅ 成功清除所有 2020 年之后的测试数据（包含 NILM 表）！数据库已彻底恢复。")

except Exception as e:
    print(f"❌ 清除失败: {e}")
finally:
    if conn:
        conn.close()