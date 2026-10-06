"""
批量删除 sys.path.append(...) 行
同时删除前面的 import sys, import os 等不再需要的 import
"""
import os
import re

src_dir = r"E:\智能体开发学习\智能体项目\体态矫正和健身计划规划智能体-手写\src"

total_files = 0
total_lines = 0

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if not file.endswith('.py'):
            continue
        
        filepath = os.path.join(root, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        removed = 0
        skip_next_import = False
        
        for i, line in enumerate(lines):
            # 匹配 sys.path.append 那一行
            if re.match(r'\s*sys\.path\.append\s*\(', line):
                removed += 1
                continue
            
            # 匹配前面的 import os, import sys, import io 等（如果只用来做 sys.path.append）
            # 简单处理：如果下一行是 sys.path.append，就跳过前面的 import
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if re.match(r'\s*sys\.path\.append\s*\(', next_line):
                    # 这行是为了 sys.path.append 准备的 import，跳过
                    if re.match(r'\s*import\s+(os|sys|io)\s*$', line):
                        continue
            
            new_lines.append(line)
        
        if removed > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            total_files += 1
            total_lines += removed
            print(f"  {os.path.relpath(filepath, src_dir)}: 删除了 {removed} 行")

print(f"\n完成！共处理 {total_files} 个文件，删除了 {total_lines} 行 sys.path.append")
