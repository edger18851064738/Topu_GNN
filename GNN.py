#!/usr/bin/env python3
"""
🚨 立即修复JSON序列化错误
直接运行此脚本即可修复demo_MAGEC.py
"""

import os
import shutil

def fix_json_error():
    """立即修复JSON序列化错误"""
    
    print("🔧 正在修复demo_MAGEC.py中的JSON序列化问题...")
    
    if not os.path.exists('demo_MAGEC.py'):
        print("❌ 找不到demo_MAGEC.py文件")
        return
    
    # 备份原文件
    shutil.copy('demo_MAGEC.py', 'demo_MAGEC.py.backup')
    print("📁 原文件已备份为: demo_MAGEC.py.backup")
    
    # 读取文件
    with open('demo_MAGEC.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在合适位置添加修复函数
    fix_function = '''
def make_json_safe(obj):
    """转换对象为JSON安全格式，处理tuple键问题"""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    else:
        return obj

'''
    
    # 在logger配置后插入
    insert_marker = "logger = logging.getLogger(__name__)"
    insert_pos = content.find(insert_marker)
    if insert_pos != -1:
        insert_pos = content.find('\n', insert_pos) + 1
        content = content[:insert_pos] + fix_function + content[insert_pos:]
    else:
        # 备用插入位置
        insert_marker = "logging.basicConfig(level=logging.INFO)"
        insert_pos = content.find(insert_marker)
        if insert_pos != -1:
            insert_pos = content.find('\n', insert_pos) + 1
            content = content[:insert_pos] + fix_function + content[insert_pos:]
    
    # 修复JSON保存调用
    old_pattern = """json.dump({
                'input_config': config,
                'training_config': training_config,
                'final_metrics': final_metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2, ensure_ascii=False, default=str)"""
    
    new_pattern = """# 修复tuple键JSON序列化问题
            safe_config = make_json_safe({
                'input_config': config,
                'training_config': training_config,
                'final_metrics': final_metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            json.dump(safe_config, f, indent=2, ensure_ascii=False, default=str)"""
    
    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)
        print("✅ 找到并修复了JSON保存代码")
    else:
        # 更宽松的匹配
        import re
        pattern = r'json\.dump\(\s*\{\s*[\'"]input_config[\'"].*?\}, f, indent=2, ensure_ascii=False, default=str\)'
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, 
                '''# 修复tuple键JSON序列化问题
            safe_config = make_json_safe({
                'input_config': config,
                'training_config': training_config,
                'final_metrics': final_metrics,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
            json.dump(safe_config, f, indent=2, ensure_ascii=False, default=str)''',
                content, flags=re.DOTALL)
            print("✅ 使用模式匹配修复了JSON保存代码")
        else:
            print("⚠️ 未找到精确的JSON保存代码，添加了修复函数")
    
    # 保存修复后的文件
    with open('demo_MAGEC.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 修复完成！")
    print("🚀 现在可以重新运行: python demo_MAGEC.py")
    print("📋 如果还有问题，请检查第1560行附近的json.dump调用")

if __name__ == "__main__":
    fix_json_error()