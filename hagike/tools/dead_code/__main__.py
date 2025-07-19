"""
死代码检测工具的命令行入口
"""

from .cli import main

if __name__ == '__main__':
    import sys
    sys.exit(main())