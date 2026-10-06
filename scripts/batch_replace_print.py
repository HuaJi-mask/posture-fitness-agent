"""
批量替换 print → logger.info
在每个文件顶部加 logger 导入
"""
import os
import re

# 需要处理的文件列表（src 目录下所有 .py 文件）
src_dir = r"E:\智能体开发学习\智能体项目\体态矫正和健身计划规划智能体-手写\src"

# 要排除的文件
exclude_files = ["logger.py"]

def process_file(filepath):
    """处理单个文件"""
    filename = os.path.basename(filepath)
    if filename in exclude_files:
        return 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 统计 print 数量
    print_count = len(re.findall(r'\bprint\s*\(', content))
    if print_count == 0:
        return 0
    
    # 检查是否已经导入了 logger
    if 'from src.utils.logger import' in content or 'from src.utils.logger import get_logger' in content:
        # 已经导入了，只替换 print
        new_content = re.sub(r'\bprint\s*\(', 'logger.info(', content)
    else:
        # 需要加导入，同时替换 print
        # 找一个合适的位置插入导入（在第一个 from import 之后）
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                insert_idx = i + 1
        
        # 加 logger 导入
        logger_import = "\nfrom src.utils.logger import get_logger\nlogger = get_logger(__name__)\n"
        lines.insert(insert_idx, logger_import)
        
        content = '\n'.join(lines)
        
        # 替换 print
        new_content = re.sub(r'\bprint\s*\(', 'logger.info(', content)
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return print_count


# 遍历所有文件
total_files = 0
total_prints = 0

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            count = process_file(filepath)
            if count > 0:
                total_files += 1
                total_prints += count
                print(f"  {os.path.relpath(filepath, src_dir)}: 替换了 {count} 个 print")

print(f"\n完成！共处理 {total_files} 个文件，替换了 {total_prints} 个 print")
