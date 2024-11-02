"""
**`matlab` 引擎管理器** \n
此处说明各种 `matlab` 操作对应的使用方式，假设引擎命名为 `m` \n
直接调用函数或创建实例：\n
result = m.function(arg1, ...) \n
obj = m.class(arg1, ...) \n
调用函数句柄：\n
value = m.feval(methodName, arg1, ...) \n
获取对象成员属性：\n
value = m.subsref(obj, m.substruct('.', propName)) \n
调用类函数：\n
value = m.feval(methodName, obj, arg1, ...) \n
调用特殊字：\n
grid on; \n
m.feval('grid', 'on') \n
"""


import matlab.engine
from hagike.utils.message import add_msg, MsgLevel
from .scripts import *
from hagike.utils.enum import *
import warnings


@advanced_enum()
class MCallType(SuperEnum):
    """调用引擎的类型"""
    func = None
    """调用函数或创建实例或函数化封装的特殊关键字或调用函数句柄"""
    obj_value = None
    """对象成员属性"""
    obj_func = None
    """对象成员函数"""
    print = None
    """打印属性"""
    else__ = None
    """其它"""


class MEngineCallError(Exception):
    """调用类型未实现"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class MEngineCallWarning(Warning):
    pass


class MEngine:
    """`matlab` 引擎"""

    def __init__(self) -> None:
        """检查并创建或链接引擎"""
        if len(matlab.engine.find_matlab()) != 0:
            self._m = matlab.engine.connect_matlab()
        else:
            add_msg(MsgLevel.Warning,
                    f"No Shared Matlab Engine On This Computer. Creating Matlab Engine, It Takes for a While!")
            self._m = matlab.engine.start_matlab()
        self._init_conf()

    def _init_conf(self) -> None:
        """初始化配置，固定配置常用调用接口"""
        self._m.addpath(matlab_script_root)
        self._obj_value = getattr(self._m, 'subsref')
        self._obj_struct = getattr(self._m, 'substruct')
        self._func = getattr(self._m, 'm_feval')

    def __call__(self, call_type: uuid_t, script: str, *args,
                 num: int = -1, obj: Any = None) -> Any:
        """
        调用引擎 \n
        :param call_type - 调用类型 \n
        :param obj - 对象句柄，若是直接调用类型则此处填 `None` \n
        :param num - 参数数量 \n
        :param script - 函数名称 \n
        :param args - 函数参数，`matlab` 中不支持关键字传参，而只支持顺序传参 \n
        :return - 返回调用结果
        """
        MCallType.check_in_(call_type)
        if call_type == MCallType.func:
            return self.call(script, *args, num=num)
        elif call_type == MCallType.obj_value:
            return self.obj_value(obj, script)
        elif call_type == MCallType.obj_func:
            return self.obj_call(script, obj, *args, num=num)
        elif call_type == MCallType.print:
            self.print(script)
        else:
            raise MEngineCallError(f"{MCallType.get_name_(call_type)} is not implemented!!!")

    def call(self, script, *args, num: int = -1) -> Any:
        """封装直接函数调用"""
        return self._func(script, num, *args)

    def obj_value(self, obj, script) -> Any:
        """封装对象属性返回"""
        return self._obj_value(obj, self._obj_struct('.', script))

    def obj_call(self, obj, script, *args, num: int = -1) -> Any:
        """封装对象函数调用"""
        return self._func(script, num, obj, *args)

    def print(self, script) -> None:
        """封装属性打印"""
        self._func('disp', 0, script)

    @property
    def m(self) -> Any:
        """返回引擎"""
        return self._m

    def exit(self) -> None:
        """断开链接并释放资源"""
        self._m.quit()

    def __del__(self) -> None:
        """释放引擎"""
        self.exit()
