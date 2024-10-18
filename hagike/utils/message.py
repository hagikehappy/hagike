"""
description:
    日志消息
"""


from .enum import *
from colorama import Fore, Back, Style


@advanced_enum()
class MsgLevel(SuperEnum):
    Run = Fore.GREEN + "RUN: " + Style.RESET_ALL
    Warning = Fore.YELLOW + "WARNING: " + Style.RESET_ALL
    Error = Fore.RED + "ERROR: " + Style.RESET_ALL
    Panic = Fore.RED + "PANIC: " + Style.RESET_ALL


def add_msg(level: int, script: str, is_print=True):
    """添加消息"""
    if is_print:
        msg = script
        if level == MsgLevel.Run.value:
            msg = Fore.GREEN + "RUN: " + Style.RESET_ALL + script
        elif level == MsgLevel.Warning.value:
            msg = Fore.YELLOW + "WARNING: " + Style.RESET_ALL + script
        elif level == MsgLevel.Error.value:
            msg = Fore.RED + "ERROR: " + Style.RESET_ALL + script
        print(msg)


def error_proc(is_exit=True):
    """错误处理"""
    if is_exit:
        exit(-1)

