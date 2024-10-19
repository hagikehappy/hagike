"""
配置类
"""


import os
import json
from .message import *
from typing import Any


def read_json(json_path: str) -> dict:
    """读取json文件并转换为字典"""
    data = dict()
    try:
        with open(json_path, 'r') as config_file:
            try:
                data = json.load(config_file)
            except json.JSONDecodeError:
                add_msg(MsgLevel.Error.value, f"read_json - File '{json_path}' Not Json Code")
                error_proc()
    except FileNotFoundError:
        add_msg(MsgLevel.Error.value, f"read_json - File '{json_path}' Not Found")
        error_proc()
    return data


def write_json(json_path: str, data: dict, settings: dict | None = None) -> None:
    """写入json文件并转换为字典"""
    dir_name = os.path.dirname(json_path)
    os.makedirs(dir_name, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

