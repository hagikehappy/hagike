"""
死代码检测器主要功能实现
"""

import os
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .analyzer import CodeAnalyzer, ProjectAnalyzer


@dataclass
class DeadCodeReport:
    """死代码报告"""
    unused_imports: Dict[str, List[Tuple[str, int]]]  # 文件 -> [(导入名, 行号)]
    unused_definitions: Dict[str, List[Tuple[str, int]]]  # 文件 -> [(定义名, 行号)]
    unused_files: List[str]  # 未使用的文件列表
    
    def is_clean(self) -> bool:
        """检查是否没有死代码"""
        return (not self.unused_imports and 
                not self.unused_definitions and 
                not self.unused_files)
    
    def total_issues(self) -> int:
        """总问题数"""
        import_count = sum(len(imports) for imports in self.unused_imports.values())
        def_count = sum(len(defs) for defs in self.unused_definitions.values())
        file_count = len(self.unused_files)
        return import_count + def_count + file_count


class DeadCodeDetector:
    """死代码检测器"""
    
    def __init__(self, project_root: str, exclude_patterns: Optional[List[str]] = None):
        """
        初始化检测器
        
        Args:
            project_root: 项目根目录
            exclude_patterns: 排除的文件模式列表
        """
        self.project_root = Path(project_root)
        self.exclude_patterns = exclude_patterns or [
            '__pycache__',
            '.git',
            '.pytest_cache', 
            'build',
            'dist',
            '*.egg-info'
        ]
        
    def should_exclude(self, file_path: Path) -> bool:
        """检查文件是否应该被排除"""
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        return False
    
    def detect_unused_imports(self, files: Optional[List[Path]] = None) -> Dict[str, List[Tuple[str, int]]]:
        """检测未使用的导入"""
        if files is None:
            analyzer = ProjectAnalyzer(str(self.project_root))
            files = analyzer.find_python_files()
        
        unused_imports = {}
        
        for file_path in files:
            if self.should_exclude(file_path):
                continue
            
            analyzer = ProjectAnalyzer(str(self.project_root))
            file_analyzer = analyzer.analyze_file(file_path)
            unused = file_analyzer.get_unused_imports()
            
            if unused:
                relative_path = str(file_path.relative_to(self.project_root))
                unused_imports[relative_path] = unused
        
        return unused_imports
    
    def detect_unused_definitions(self, files: Optional[List[Path]] = None) -> Dict[str, List[Tuple[str, int]]]:
        """检测未使用的定义"""
        if files is None:
            analyzer = ProjectAnalyzer(str(self.project_root))
            files = analyzer.find_python_files()
        
        unused_definitions = {}
        
        for file_path in files:
            if self.should_exclude(file_path):
                continue
            
            analyzer = ProjectAnalyzer(str(self.project_root))
            file_analyzer = analyzer.analyze_file(file_path)
            unused = file_analyzer.get_unused_definitions()
            
            if unused:
                relative_path = str(file_path.relative_to(self.project_root))
                unused_definitions[relative_path] = unused
        
        return unused_definitions
    
    def detect_unused_files(self) -> List[str]:
        """检测未使用的文件"""
        analyzer = ProjectAnalyzer(str(self.project_root))
        analyzer.find_python_files()
        return analyzer.find_unused_files()
    
    def detect_all(self) -> DeadCodeReport:
        """检测所有类型的死代码"""
        analyzer = ProjectAnalyzer(str(self.project_root))
        files = analyzer.find_python_files()
        
        # 过滤掉需要排除的文件
        files = [f for f in files if not self.should_exclude(f)]
        
        unused_imports = self.detect_unused_imports(files)
        unused_definitions = self.detect_unused_definitions(files)
        unused_files = self.detect_unused_files()
        
        return DeadCodeReport(
            unused_imports=unused_imports,
            unused_definitions=unused_definitions, 
            unused_files=unused_files
        )
    
    def print_report(self, report: DeadCodeReport) -> None:
        """打印检测报告"""
        if report.is_clean():
            print("✅ 没有发现死代码！")
            return
        
        print(f"🔍 发现 {report.total_issues()} 个死代码问题：\n")
        
        # 未使用的导入
        if report.unused_imports:
            print("📦 未使用的导入：")
            for file_path, imports in report.unused_imports.items():
                print(f"  📁 {file_path}:")
                for name, line in imports:
                    print(f"    - 行 {line}: {name}")
            print()
        
        # 未使用的定义
        if report.unused_definitions:
            print("🔧 未使用的定义：")
            for file_path, definitions in report.unused_definitions.items():
                print(f"  📁 {file_path}:")
                for name, line in definitions:
                    print(f"    - 行 {line}: {name}")
            print()
        
        # 未使用的文件
        if report.unused_files:
            print("📄 未使用的文件：")
            for file_path in report.unused_files:
                print(f"  - {file_path}")
            print()


def detect_dead_code(project_root: str, exclude_patterns: Optional[List[str]] = None) -> DeadCodeReport:
    """便捷函数：检测项目中的死代码"""
    detector = DeadCodeDetector(project_root, exclude_patterns)
    return detector.detect_all()


def print_dead_code_report(project_root: str, exclude_patterns: Optional[List[str]] = None) -> None:
    """便捷函数：检测并打印死代码报告"""
    detector = DeadCodeDetector(project_root, exclude_patterns)
    report = detector.detect_all()
    detector.print_report(report)