"""
AST分析器，用于分析Python代码的使用情况
"""

import ast
import os
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path


class CodeAnalyzer(ast.NodeVisitor):
    """代码分析器，使用AST分析Python代码"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports: Dict[str, int] = {}  # 导入的名称及其行号
        self.definitions: Dict[str, int] = {}  # 定义的函数/类及其行号
        self.usages: Set[str] = set()  # 使用的名称
        self.from_imports: Dict[str, Dict[str, int]] = {}  # from import语句
        
    def analyze(self, source: str) -> None:
        """分析源代码"""
        try:
            tree = ast.parse(source, filename=self.file_path)
            self.visit(tree)
        except SyntaxError as e:
            print(f"Warning: Syntax error in {self.file_path}: {e}")
    
    def visit_Import(self, node: ast.Import) -> None:
        """访问import语句"""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = node.lineno
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """访问from import语句"""
        if node.module:
            if node.module not in self.from_imports:
                self.from_imports[node.module] = {}
            for alias in node.names:
                if alias.name == '*':
                    # 对于from module import *，我们无法精确追踪
                    continue
                name = alias.asname if alias.asname else alias.name
                self.from_imports[node.module][name] = node.lineno
                # 也记录在imports中方便统一处理
                self.imports[name] = node.lineno
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """访问函数定义"""
        if not node.name.startswith('_'):  # 不检查私有函数
            self.definitions[node.name] = node.lineno
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """访问异步函数定义"""
        if not node.name.startswith('_'):  # 不检查私有函数
            self.definitions[node.name] = node.lineno
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """访问类定义"""
        if not node.name.startswith('_'):  # 不检查私有类
            self.definitions[node.name] = node.lineno
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name) -> None:
        """访问名称节点"""
        if isinstance(node.ctx, ast.Load):
            self.usages.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """访问属性节点"""
        # 对于 module.function 这种调用，我们也需要记录
        if isinstance(node.value, ast.Name):
            self.usages.add(node.value.id)
        self.generic_visit(node)
    
    def get_unused_imports(self) -> List[Tuple[str, int]]:
        """获取未使用的导入"""
        unused = []
        for name, line in self.imports.items():
            if name not in self.usages:
                unused.append((name, line))
        return unused
    
    def get_unused_definitions(self) -> List[Tuple[str, int]]:
        """获取未使用的定义"""
        unused = []
        for name, line in self.definitions.items():
            if name not in self.usages:
                unused.append((name, line))
        return unused


class ProjectAnalyzer:
    """项目级分析器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.python_files: List[Path] = []
        self.file_imports: Dict[str, Set[str]] = {}  # 文件导入关系
        self.all_definitions: Dict[str, Set[str]] = {}  # 所有定义
        self.all_usages: Set[str] = set()  # 所有使用
        
    def find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 排除常见的忽略目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        self.python_files = python_files
        return python_files
    
    def analyze_file(self, file_path: Path) -> CodeAnalyzer:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            analyzer = CodeAnalyzer(str(file_path))
            analyzer.analyze(source)
            
            # 记录文件级信息
            relative_path = str(file_path.relative_to(self.project_root))
            self.all_definitions[relative_path] = set(analyzer.definitions.keys())
            self.all_usages.update(analyzer.usages)
            
            return analyzer
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return CodeAnalyzer(str(file_path))
    
    def get_import_graph(self) -> Dict[str, Set[str]]:
        """获取导入关系图"""
        import_graph = {}
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                imports = set()
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])
                
                relative_path = str(file_path.relative_to(self.project_root))
                import_graph[relative_path] = imports
                
            except Exception as e:
                print(f"Error analyzing imports in {file_path}: {e}")
                relative_path = str(file_path.relative_to(self.project_root))
                import_graph[relative_path] = set()
        
        return import_graph
    
    def find_unused_files(self) -> List[str]:
        """查找未被引用的文件"""
        # 这是一个简化的实现，实际情况会更复杂
        import_graph = self.get_import_graph()
        all_imports = set()
        
        # 收集所有被导入的模块
        for imports in import_graph.values():
            all_imports.update(imports)
        
        unused_files = []
        for file_path in self.python_files:
            relative_path = str(file_path.relative_to(self.project_root))
            
            # 跳过特殊文件
            if (file_path.name == '__init__.py' or 
                file_path.name.startswith('test_') or
                '/tests/' in relative_path):
                continue
            
            # 检查文件是否被导入
            module_name = relative_path.replace('/', '.').replace('.py', '')
            parts = module_name.split('.')
            
            # 检查是否有任何部分被导入
            is_imported = False
            for i in range(len(parts)):
                partial_name = '.'.join(parts[:i+1])
                if partial_name in all_imports:
                    is_imported = True
                    break
            
            if not is_imported:
                unused_files.append(relative_path)
        
        return unused_files