"""
势能面搜索算法交互式可视化教学工具 - 入口文件

运行方式：
    conda activate torch
    python main.py
"""

import sys
import os

# 添加本地libs目录到Python路径（包含PyQt5等依赖）
# 注意：只添加libs中venv没有的包，避免覆盖venv中已安装的包
# 优先查找当前目录下的libs，其次查找上级目录
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_cur_dir)
libs_dir = None
for d in [os.path.join(_cur_dir, 'libs'), os.path.join(_parent_dir, 'libs')]:
    if os.path.isdir(d):
        libs_dir = d
        break
if libs_dir:
    # 使用append而非insert，让venv中的包优先
    sys.path.append(libs_dir)

# 设置Qt平台插件路径（pip --target安装时需要）
import PyQt5
qt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_plugin_path, 'platforms')

from app import main

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        input("按回车键退出...")
