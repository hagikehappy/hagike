"""
日志消息
"""


from .enum import *
from colorama import Fore, Back, Style


@advanced_enum()
class MsgLevel(SuperEnum):
    Run = Fore.GREEN + "RUN: "
    Warning = Fore.YELLOW + "WARNING: "
    Error = Fore.RED + "ERROR: "
    Panic = Fore.RED + "PANIC: "


def add_msg(level: int, script: str, is_print=True):
    """添加消息"""
    if is_print:
        msg = script
        MsgLevel.check_in_(level)
        msg = MsgLevel.get_value_(level) + msg + Style.RESET_ALL
        print(msg)


def error_proc(is_exit=True):
    """错误处理"""
    if is_exit:
        exit(-1)

