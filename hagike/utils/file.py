"""
***与文件处理相关的库函数*** \n
"""


import os


class FileWritableError(Exception):
    """文件不可写异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class FileReadableError(Exception):
    """文件不可读异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


def check_path_readable(path: str, is_raise: bool = True) -> bool:
    """
    确保目标可读取，检查项包括： \n
    0. 检查路径本身的合法性 \n
    1. 路径本身存在 \n
    2. 路径是文件而非文件夹 \n
    3. 路径有读取权限 \n
    """
    # 0
    try:
        os.path.dirname(path)
    except TypeError:
        if is_raise:
            raise FileReadableError(f"ERROR: {path} is not a string!!!")
        else:
            return False
    # 1
    if not os.path.exists(path):
        if is_raise:
            raise FileReadableError(f"ERROR: {path} does not exist!!!")
        else:
            return False
    # 2
    if os.path.isdir(path):
        if is_raise:
            raise FileReadableError(f"ERROR: {path} is a dir!!!")
        else:
            return False
    # 3
    if os.access(path, os.R_OK):
        return True
    else:
        if is_raise:
            raise FileReadableError(f"ERROR: {path} is not Readable, Permission Denied!!!")
        else:
            return False


def ensure_path_writable(path: str, is_raise: bool = True) -> bool:
    """
    确保目标路径可写入，包括检查项： \n
    0. 检查路径本身的合法性 \n
    1. 路径本身不是一个已存在的文件夹 \n
    2. 确保路径的父文件夹存在，若不存在则创建 \n
    3. 确保其有写入权限 \n
    如果指定 `is_raise=True`，则会在不可达时触发异常，否则仅返回 `False`
    """
    # 0
    try:
        directory = os.path.dirname(path)
    except TypeError:
        if is_raise:
            raise FileWritableError(f"ERROR: {path} is not a string!!!")
        else:
            return False
    # 1
    if os.path.isdir(path):
        if is_raise:
            raise FileWritableError(f"ERROR: {path} is a dir!!!")
        else:
            return False
    # 2
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError:
        if is_raise:
            raise FileWritableError(f"ERROR: dir {directory} is cannot be created, Permission Denied!!!")
        else:
            return False
    # 3
    if os.access(directory, os.W_OK):
        return True
    else:
        if is_raise:
            raise FileWritableError(f"ERROR: dir {directory} is not Writable, Permission Denied!!!")
        else:
            return False




