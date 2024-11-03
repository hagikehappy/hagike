"""
系统函数类
"""


from __future__ import annotations
from ..utils import *
from typing import List, Dict, Any
from .const import *
from .error import *
import matplotlib.pyplot as plt
import numpy as np
from ..engine.engine import *


class MSys:
    """`matlab` 系统"""

    def __init__(self, m: MEngine,
                 num: List[float] | None = None, den: List[float] | None = None,
                 z: List[float] | None = None, p: List[float] | None = None, k: float | None = None,
                 fb: MSys | None = None, tf: Any = None) -> None:
        """
        初始化参数，有三种输入方式（直接输入、输入零极点、输入分子分母系数），可选是否进行反馈 \n
        :param m - `matlab` 引擎 \n
        :param tf - 若直接给出 `matlab` 中的 `tf` 系统，则忽略系数项，直接指定
        :param num - 系统函数的分子系数，降幂排列 \n
        :param den - 系统函数的分母系数，降幂排列 \n
        :param z - 零点
        :param p - 极点
        :param k - 增益
        :param fb - 也需要为 `Msys` 类型 \n
        """
        self._m, self._fb = m, fb
        # 获取开环系统
        if tf is not None:
            self._G = tf
            num = np.array(m.obj_value(self._G, 'num'))[0][0]
            den = np.array(m.obj_value(self._G, 'den'))[0][0]
        elif num is not None and den is not None:
            self._G = m.call('tf', m_double(num), m_double(den), num=1)
        elif z is not None and p is not None and k is not None:
            num, den = m.call('zp2tf', m_double(z), m_double(p), m_double(k), num=2)
            num, den = np.array(num)[0], np.array(den)[0]
            self._G = m.call('tf', m_double(num), m_double(den), num=1)
        else:
            raise MSysInputError(f"Incorrect Input Way!!!")

        # 获取闭环系统
        if fb is None:
            self._H = self._G
            self._num, self._den = num, den
        else:
            self._H = m.call('feedback', self._G, fb.H, num=1)
            # 返回类型为 matlab.double 类型，需要将其解包为 numpy 格式
            self._num = np.array(m.obj_value(self._H, 'num'))[0][0]
            self._den = np.array(m.obj_value(self._H, 'den'))[0][0]
        # 获取系统增益
        self._amp = self._num[-1] / self._den[-1] if self._den[-1] != 0.0 else None
        self._t, self._response, self._input_t = None, None, None

    @property   # 返回 `matlab` 系统
    def H(self) -> Any: return self._H
    @property   # 返回系统稳态增益
    def amp(self) -> float | None: return self._amp
    @property   # 返回响应
    def response(self) -> List[float] | None: return self._response

    def _check_response(self, input_t: uuid_t | None = None) -> None:
        """检查是否已经生成指定类型 `response`，若没有则报错"""
        if self._input_t is None:
            raise MSysResponseError(f"No Response Generated Yet!!!")
        if input_t is not None and self._input_t != input_t:
            raise MSysResponseError(
                f"Response {MSysInput.get_name_(self._input_t)} is not "
                f"the asked type {MSysInput.get_name_(input_t)}!!!")

    def gen_response(self, t: List[float], input_t: uuid_t, input_s: List[float] | None = None) -> List[float]:
        """
        生成响应，结果记录在实例中 \n
        :param t - 时刻表 \n
        :param input_t - 输入类型 \n
        :param input_s - 输入系统，当且仅当 `input_t` 为 `else__` 时才有效 \n
        :return 系统响应 \n
        """
        self._t, self._input_t = t, input_t
        MSysInput.check_in_(input_t)
        if input_t == MSysInput.step:
            self._response = self._m.call('step', self._H, m_double(t), num=1)
        elif input_t == MSysInput.impulse:
            self._response = self._m.call('impulse', self._H, m_double(t), num=1)
        elif input_t == MSysInput.ramp:
            self._response = self._m.call('lsim', self._H, m_double(t), m_double(t), num=1)
        elif input_t == MSysInput.else__:
            self._response = self._m.call('lsim', self._H, m_double(input_s), m_double(t), num=1)
        else:
            raise MSysInputError(f"{MSysInput.get_name_(input_t)} is not implemented!!!")
        return self._response

    def plot_response(self, para: Dict[str, Any] | None = None, save_path: str | None = None):
        """绘制响应图，`para` 为绘图配置项"""
        self._check_response()
        if para is None:
            para = dict()
        plt.plot(self._t, self._response, **para)
        plt.grid('on')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def overshoot(self) -> float:
        """返回超调量"""
        self._check_response(MSysInput.step)
        return self._m.call('overshoot', m_double(self._response), num=1)

    def settlingtime(self, delta: float = 0.05) -> float | None:
        """
        返回稳态响应时间，若响应中未达到稳态条件则返回 `None` \n
        这里假定响应已经达到类稳态状态，使用方法为从后向前扫描响应 \n
        :param delta - 容差 \n
        """
        self._check_response(MSysInput.step)
        if self._amp is None:
            return None
        r_map = np.abs(np.array(self._response) / self.amp - 1) > delta
        r_len = len(r_map)
        first_settle = r_len - np.argmax(~r_map)
        if first_settle == r_len:
            return None
        else:
            return self._t[first_settle]


class MSysConst(MSys):
    """`matlab` 常数块"""
    def __init__(self, m: Any, amp: float = 1.0):
        """创建常数系统"""
        super().__init__(m, [float(amp)], [1.0])

