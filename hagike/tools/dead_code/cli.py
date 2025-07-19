"""
死代码检测命令行接口
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from .detector import DeadCodeDetector


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='检测Python项目中的死代码（未使用的导入、函数、类和文件）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m hagike.tools.dead_code .                    # 检测当前目录
  python -m hagike.tools.dead_code /path/to/project     # 检测指定项目
  python -m hagike.tools.dead_code . --exclude tests    # 排除tests目录
  python -m hagike.tools.dead_code . --imports-only     # 只检测未使用的导入
        """
    )
    
    parser.add_argument(
        'project_root',
        nargs='?',
        default='.',
        help='项目根目录路径 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '--exclude',
        nargs='*',
        default=[],
        help='排除的目录或文件模式'
    )
    
    parser.add_argument(
        '--imports-only',
        action='store_true',
        help='仅检测未使用的导入'
    )
    
    parser.add_argument(
        '--definitions-only', 
        action='store_true',
        help='仅检测未使用的定义（函数、类）'
    )
    
    parser.add_argument(
        '--files-only',
        action='store_true',
        help='仅检测未使用的文件'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='以JSON格式输出结果'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式，只有发现问题时才输出'
    )
    
    return parser


def main(args: Optional[List[str]] = None) -> int:
    """主函数"""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # 检查项目根目录
    project_root = Path(parsed_args.project_root)
    if not project_root.exists():
        print(f"错误：项目目录 '{project_root}' 不存在", file=sys.stderr)
        return 1
    
    if not project_root.is_dir():
        print(f"错误：'{project_root}' 不是一个目录", file=sys.stderr)
        return 1
    
    # 设置排除模式
    exclude_patterns = [
        '__pycache__',
        '.git', 
        '.pytest_cache',
        'build',
        'dist',
        '*.egg-info'
    ]
    exclude_patterns.extend(parsed_args.exclude)
    
    # 创建检测器
    detector = DeadCodeDetector(str(project_root), exclude_patterns)
    
    try:
        # 根据参数选择检测类型
        if parsed_args.imports_only:
            unused_imports = detector.detect_unused_imports()
            from .detector import DeadCodeReport
            report = DeadCodeReport(unused_imports, {}, [])
        elif parsed_args.definitions_only:
            unused_definitions = detector.detect_unused_definitions()
            from .detector import DeadCodeReport
            report = DeadCodeReport({}, unused_definitions, [])
        elif parsed_args.files_only:
            unused_files = detector.detect_unused_files()
            from .detector import DeadCodeReport
            report = DeadCodeReport({}, {}, unused_files)
        else:
            report = detector.detect_all()
        
        # 输出结果
        if parsed_args.json:
            import json
            output = {
                'unused_imports': report.unused_imports,
                'unused_definitions': report.unused_definitions,
                'unused_files': report.unused_files,
                'total_issues': report.total_issues()
            }
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if not parsed_args.quiet or not report.is_clean():
                detector.print_report(report)
        
        # 返回退出码
        return 0 if report.is_clean() else 1
        
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1


def cli():
    """命令行入口点"""
    sys.exit(main())


if __name__ == '__main__':
    cli()