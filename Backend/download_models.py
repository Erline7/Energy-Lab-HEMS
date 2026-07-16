from modelscope import snapshot_download
import os
import shutil

print("Starting clean download...")

# 目标路径
target_dir = os.path.abspath("local_models/Qwen/Qwen3.5-2B")

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

# 开始下载
model_dir = snapshot_download(
    'Qwen/Qwen3.5-2B")',
    cache_dir=os.path.abspath("local_models/temp_cache") 
)

if not os.path.exists(target_dir):
    shutil.copytree(model_dir, target_dir)

print(f"Success! Model files are now in: {target_dir}")