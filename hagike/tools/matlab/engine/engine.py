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
from hagike.tools.matlab.scripts import *
from .error import *
from .const import *


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
        self._operator = getattr(self._m, 'm_operator')

    def __call__(self, call_type: uuid_t, script: str | uuid_t, *args,
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
        MCall.check_in_(call_type)
        if call_type == MCall.func:
            return self.call(script, *args, num=num)
        elif call_type == MCall.obj_value:
            return self.obj_value(obj, script)
        elif call_type == MCall.obj_func:
            return self.obj_call(script, obj, *args, num=num)
        elif call_type == MCall.print:
            self.print(script, *args)
        elif call_type == MCall.operator:
            return self.operator(script, *args)
        else:
            raise MEngineCallError(f"{MCall.get_name_(call_type)} is not implemented!!!")

    def call(self, script: str, *args, num: int = -1) -> Any:
        """封装直接函数调用"""
        return self._func(script, num, *args)

    def obj_value(self, obj: Any, script: str) -> Any:
        """封装对象属性返回"""
        return self._obj_value(obj, self._obj_struct('.', script))

    def obj_call(self, obj, script: str, *args, num: int = -1) -> Any:
        """封装对象函数调用"""
        return self._func(script, num, obj, *args)

    def print(self, *args) -> None:
        """封装属性打印"""
        self._func('disp', 0, *args)

    def operator(self, script: uuid_t, *arg):
        """封装运算符"""
        base_cls = MOperator.get_base_(script)
        op_type = MOperator.get_name_(base_cls.get_uuid_())
        func = base_cls.get_value_(script)
        return self._operator(op_type, func, *arg)

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
