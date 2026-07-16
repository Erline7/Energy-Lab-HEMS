import sqlite3
import pandas as pd

# 连接数据库
conn = sqlite3.connect('hems_data.db')

# 查看 telemetry_main 表里有多少条数据
df = pd.read_sql_query("SELECT count(*) as count, min(timestamp) as min_ts, max(timestamp) as max_ts FROM telemetry_main", conn)
print(f"telemetry_main 表状态：{df.to_dict('records')}")

# 如果上面查不到，看看别的表有没有
df_water = pd.read_sql_query("SELECT count(*) as count FROM water_main_logs", conn)
print(f"water_main_logs 表状态：{df_water.to_dict('records')}")

conn.close()