"""
测试死代码检测功能
"""

import os
import tempfile
import shutil
from pathlib import Path
from hagike.tools.dead_code import DeadCodeDetector, detect_dead_code


def test_unused_imports():
    """测试未使用导入检测"""
    # 创建临时项目
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # 创建测试文件
        test_file = project_root / "test_unused_imports.py"
        test_file.write_text("""
import os
import sys
from typing import Dict
import json

def hello():
    print("Hello")
    return sys.version
""")
        
        detector = DeadCodeDetector(str(project_root))
        report = detector.detect_all()
        
        # 应该检测到未使用的导入
        unused_imports = report.unused_imports.get("test_unused_imports.py", [])
        unused_names = [name for name, line in unused_imports]
        
        assert "os" in unused_names
        assert "Dict" in unused_names 
        assert "json" in unused_names
        assert "sys" not in unused_names  # sys被使用了


def test_unused_definitions():
    """测试未使用定义检测"""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        test_file = project_root / "test_unused_defs.py"
        test_file.write_text("""
def used_function():
    return "used"

def unused_function():
    return "unused"

class UsedClass:
    pass

class UnusedClass:
    pass

result = used_function()
instance = UsedClass()
""")
        
        detector = DeadCodeDetector(str(project_root))
        report = detector.detect_all()
        
        unused_defs = report.unused_definitions.get("test_unused_defs.py", [])
        unused_names = [name for name, line in unused_defs]
        
        assert "unused_function" in unused_names
        assert "UnusedClass" in unused_names
        assert "used_function" not in unused_names
        assert "UsedClass" not in unused_names


def test_exclude_patterns():
    """测试排除模式功能"""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # 创建应该被排除的文件
        test_dir = project_root / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_something.py"
        test_file.write_text("import unused_import")
        
        detector = DeadCodeDetector(str(project_root), exclude_patterns=["tests"])
        report = detector.detect_all()
        
        # tests目录下的文件应该被排除
        assert "tests/test_something.py" not in report.unused_imports


def test_convenience_function():
    """测试便捷函数"""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        test_file = project_root / "simple.py"
        test_file.write_text("import unused")
        
        report = detect_dead_code(str(project_root))
        assert not report.is_clean()
        assert report.total_issues() > 0


if __name__ == "__main__":
    test_unused_imports()
    test_unused_definitions()
    test_exclude_patterns()
    test_convenience_function()
    print("✅ 所有测试通过！")