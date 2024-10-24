"""
***高级Enum类*** \n
**term**: \n
    SuperEnum - 高级枚举类的父类模板
    advanced_enum - 根级的装饰器，用于自动生成枚举类的各项配置
    uuid - 唯一标识符
    index - 索引
    name - 名称
    value - 值
**notice**: \n
    对于枚举成员：
    1. 本身的值在定义时是value，在访问时是uuid；对于部分IDE的代码高亮，为消除格式不符警告，可以类型标注为uuid_t
    2. index不应该被外部访问，而仅作为迭代器的索引值
    3. 命名不能以'__'开头，否则会被忽略
    4. 值如果是继承了SuperEnum的枚举类型，则会递归导入，需要确保此处枚举类的归属是唯一的，否则uuid会被多次修改；
    5. 如果未启用顺序访问索引，无法保证枚举类的书写顺序就是index顺序，顺序是由Python机制决定的（默认是按名称顺序）
    6. '_xxx'方式命名的枚举量为内置属性，不可用常规枚举量占用，且只能配置其中的配置字部分，否则会报错
    7. 其它的外显的普通枚举量可以命名为任意的'xxx'，前后不应该带有下划线!!!
    8. 对于'xxx__'方式命名的枚举量，这里会将其作为隐藏枚举量，不会被包含在index列表中，也不会分配index值
    9. 所有父类方法命名为'xxx_'，以此与隐藏枚举量进行区分
    10. 根类的值是会被包含在uuid及其映射中的，但由于其没有父类，因而不会被包含在任何index列表中，其index本身总为0
.. todo::
    待添加实例化常量表为本地配置表的功能； \n
    待添加从文件中加载 / 赋值常量表的功能； \n
    待添加对 `uuid_t` 类型的封装，使得封装内可以带有Enum类标识符，防止误输入属于另一个Enum类的标识符而绕过检查
"""

from dataclasses import dataclass
from typing import Any, List, Tuple, Dict, Sequence, Set, Iterator, Mapping
from copy import deepcopy

# 重定义Enum中的标识符类型
uuid_t = int
index_t = int

_enum_conf_word = ('_value', '_sequence', '_hide', '_blank')
"""
SuperEnum类型配置字，在父类SuperEnum中定义的值为默认值 \n

:param _value:  \n
    对于group本身的赋值需要写在成员_value中，否则会被视为None，访问时依然通过value \n
:param _sequence:  \n
    如果在某子类下启用顺序访问索引，则需要赋值成员_sequence: Tuple[str]； \n
    其中需要按顺序列出所有成员名称；如果未列全或有不存在的成员名称则在初始化时会报错 \n
:param _hide: \n
    是否将类本身的值作为隐藏枚举值 \n
:param _blank:  \n
    打印时的单位空格长度 \n
"""

_enum_hide_word = ('_uuid', '_pack', '_length',
                   '_index2uuid', '_uuid2pack', '_uuid2sub',
                   '_uuid_all', '_uuid_hide', '_uuid_sub')
"""
SuperEnum类型隐藏字 \n

:param _uuid: \n
    子类本身的唯一标识符 \n
:param _pack: \n
    存储子类本身的信息，是信息的打包形式 \n
:param _length: \n
    子类的非隐藏成员数量，不包括子类本身 \n
:param _index2uuid:  \n
    子类的非隐藏成员索引到唯一标识符的映射，不包括子类本身 \n
:param _uuid2pack:  \n
    子类下的唯一标识符到数据包的映射，不包括子类本身 \n
:param _uuid2sub: \n
    子类下的唯一标识符到孙类的映射 \n
:param _uuid_all: \n
    子类下所有唯一标识符的列表，其中非隐藏部分在前，隐藏部分在后 \n
:param _uuid_hide: \n
    子类下所有隐式枚举成员的集合 \n
:param _uuid_sub: \n
    子类下所有孙类枚举成员的集合 \n
"""


class EnumOccupiedError(Exception):
    """枚举类关键字占用异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class EnumSequenceError(Exception):
    """枚举类顺序访问索引异常"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class EnumTypeError(Exception):
    """枚举类的配置项的类型不正确"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class EnumUuidError(Exception):
    """枚举类的uuid不存在"""
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


@dataclass
class _EnumPack:
    """常量数据包"""
    uuid: uuid_t = None
    index: index_t = None
    name: str = None
    value: Any = None

    def print(self, is_value: bool = False) -> None:
        """打印值，is_value表示是否打印值"""
        print(f"{self.name}({self.index}) -> {self.uuid}", end='')
        if is_value:
            print(f": {self.value}", end='')
        print()


class SuperEnum:
    """Enum类的父类"""
    # 配置属性
    _value: Any | None = None
    _sequence: Tuple[str] = None
    _hide: bool = False
    _blank: int = 4
    # 隐藏属性
    _uuid: uuid_t
    _pack: _EnumPack
    _length: int
    _index2uuid: List[uuid_t]
    _uuid2pack: Dict[uuid_t, _EnumPack]
    _uuid2sub: Dict[uuid_t, Any]
    _uuid_all: List[uuid_t]
    _uuid_hide: Set[uuid_t]
    _uuid_sub: Set[uuid_t]

    @classmethod
    def get_uuid_(cls) -> uuid_t:
        """获得类本身的uuid"""
        return cls._uuid

    @classmethod
    def get_name_(cls, uuid: uuid_t) -> str:
        """获得枚举量的名称"""
        return cls._uuid2pack[uuid].name

    @classmethod
    def get_index_(cls, uuid: uuid_t) -> index_t:
        """获得枚举量的名称"""
        return cls._uuid2pack[uuid].index

    @classmethod
    def get_value_(cls, src: index_t | uuid_t, index_or_uuid: bool = False) -> Any:
        """返回深拷贝赋值，index_or_uuid决定根据index还是uuid返回值"""
        uuid = cls._index2uuid[src] if index_or_uuid else src
        pack_n = cls._uuid2pack[uuid]
        return deepcopy(pack_n.value)

    @classmethod
    def check_in_(cls, uuid: uuid_t, all_or_index: bool = False, is_raise: bool = True) -> bool:
        """检查是否uuid是否被枚举类包含，all_or_hide指定是否仅考虑显变量，is_raise指定若不存在是否引发错误"""
        if all_or_index:
            is_in = True if uuid in cls._uuid_all else False
        else:
            is_in = True if uuid in cls._index2uuid else False
        if is_raise:
            if not is_in:
                raise EnumUuidError(f"ERROR: {uuid} is not in enum!!!")
        return is_in

    @classmethod
    def check_include_(cls, enum_list: List[uuid_t], all_or_index: bool = False) -> bool:
        """检查列表内的uuid是否都包含在枚举类中，all_or_hide指定是否仅考虑显变量"""
        is_include = True
        for uuid in enum_list:
            if not cls.check_in_(uuid, all_or_index):
                is_include = False
                break
        return is_include

    @classmethod
    def dict_(cls, enum_dict: Mapping[uuid_t, Any] = None, is_force: bool = True) -> Dict[uuid_t, Any]:
        """
        填补Enum类中不在常量表部分的默认赋值，生成dict； \n
        如果选中is_force则会检查enum_dict中的key是否都在enum类的非隐藏部分中，若不满足则会报错； \n
        此项检查用于排除不正常输入的字典项，如隐藏的enum成员。 \n
        """
        if enum_dict is None:
            enum_dict = dict()
        uuid_dict = enum_dict.keys()
        for index in range(cls._length):
            uuid = cls._index2uuid[index]
            if uuid not in uuid_dict:
                enum_dict[uuid] = cls.get_value_(uuid, False)
        # 检查是否完全包含于
        if is_force:
            if cls._length != len(uuid_dict):
                raise EnumUuidError(
                    f"ERROR: dict(len={len(uuid_dict)}) is not included in enum(len={cls._length})!!!")
        return enum_dict

    @classmethod
    def list_(cls, enum_dict: Mapping[uuid_t, Any] = None, is_default: bool = False) -> List[Any]:
        """
        将dict根据index顺序进行排序，要求所有enum_dict中的key都被包含于enum类的非隐藏部分； \n
        is_default决定是否补全默认值；
        """
        enum_list = []
        if enum_dict is None:
            enum_dict = dict()
        uuid_dict = enum_dict.keys()
        # 检查是否完全包含于
        for uuid in uuid_dict:
            if uuid not in cls._index2uuid:
                raise EnumUuidError(f"ERROR: dict is not included in enum for uuid({uuid})!!!")
        for index in range(cls._length):
            uuid = cls._index2uuid[index]
            if uuid in uuid_dict:
                enum_list.append(enum_dict[uuid])
            elif is_default and uuid not in uuid_dict:
                enum_list.append(cls.get_value_(uuid, False))

        return enum_list

    @classmethod
    def iter_(cls) -> Iterator[int]:
        """返回index2uuid迭代器"""
        return iter(cls._index2uuid)

    @classmethod
    def print_pack_(cls, uuid: uuid_t, blank: int = 0, is_value: bool = False) -> None:
        """打印单个枚举量的信息"""
        blank_str = '' + ' ' * blank
        pack = cls._uuid2pack[uuid]
        print(blank_str, end='')
        pack.print(is_value)

    @classmethod
    def print_(cls, is_value: bool = False) -> None:
        """打印枚举类单级信息"""
        print()
        cls._pack.print(is_value)
        # 优先顺序打印非隐藏值
        for uuid in cls._uuid_all:
            cls.print_pack_(uuid, cls._blank, is_value)
        print()

    @classmethod
    def tree_(cls, is_value: bool = False):
        """以树形结构递归打印该枚举类信息"""

        def regress_enum(cls_n: Any, blank_n: int) -> None:
            """递归列举"""
            # 优先顺序打印非隐藏值
            for uuid_n in cls_n._uuid_all:
                cls_n.print_pack_(uuid_n, blank_n, is_value)
                if uuid_n in cls_n._uuid_sub:
                    regress_enum(cls_n._uuid2sub[uuid_n], blank_n + cls_n._blank)

        # 递归入口
        print()
        cls._pack.print(is_value)
        regress_enum(cls, cls._blank)
        print()


def advanced_enum():
    """
    该函数作为常量表的装饰器，自动建立映射，子类与子成员均视为常量，封装为常量类型，仅用于顶级Enum。
    """

    def decorator(cls):
        """装饰器，进行常量封装"""

        def check_key(keys: Sequence, all_or_hide: bool = True) -> None:
            """检查是否存在关键字冲突，all_or_hide指定全部检查还是仅检查隐藏属性"""
            for word in keys:
                if word in _enum_hide_word:
                    raise EnumOccupiedError(f"ERROR: {word} in enum_hide_word, change a Name!!!")
            if all_or_hide:
                for word in keys:
                    if word in _enum_conf_word:
                        raise EnumOccupiedError(f"ERROR: {word} in enum_conf_word, change a Name!!!")

        def check_conf(cls_n: Any) -> None:
            """检查枚举类的配置项的类型与值是否正确"""
            if not isinstance(cls_n._hide, bool):
                raise EnumTypeError(f"ERROR: _hide_ typeof {type(cls_n._hide)} but not bool!!!")
            if isinstance(cls_n._blank, int):
                if cls_n._blank < 0:
                    raise EnumTypeError(f"ERROR: _blank_({cls_n._blank}) < 0!!!")
            else:
                raise EnumTypeError(f"ERROR: _blank_ typeof {type(cls_n._blank)} but not int!!!")

        def regress_enum(uuid_n: uuid_t, cls_n: Any) -> uuid_t:
            """逐目录递归赋值uuid常量表，不会赋值顶级enum组"""
            uuid2pack_n: Dict[uuid_t, _EnumPack] = dict()
            index2uuid_n: List[uuid_t | None] = list()
            uuid_hide_n: Set[uuid_t] = set()
            uuid2sub_n: Dict[uuid_t, Any] = dict()
            index_n = 0
            # cls_n.__dict__仅会取在枚举类进行定义的变量，而dir(cls_n)会取包括SuperEnum类内的所有变量
            all_attrs_n = list(cls_n.__dict__.keys())
            all_attrs_n.reverse()  # 默认普通枚举量在前，子类在后

            # 检查是否存在关键字占用
            check_key(all_attrs_n, all_or_hide=False)
            # 检查配置字是否合法
            check_conf(cls_n)

            # 判断是否启用局部顺序映射表，如果启用则判断是否合法（是否恰好一致）并调换顺序
            seq_n, is_seq, seq_len = cls_n._sequence, False, None
            if seq_n is not None:
                is_seq, seq_len = True, len(seq_n)
                # 检查以确保_sequence_中没有关键字冲突
                check_key(seq_n, all_or_hide=True)
                index2uuid_n = [None for _ in range(seq_len)]

            for attr_n in all_attrs_n:
                # 排除魔法属性，'_value'在父级中设置，不在本级设置
                if attr_n.startswith('__'):
                    continue
                # 排除内置属性，并确保子类不存在自定义的内置属性
                elif attr_n.startswith('_'):
                    if attr_n not in _enum_conf_word:
                        raise EnumOccupiedError(f"ERROR: {attr_n} not in enum_conf_word, change a Name!!!")
                # 检查枚举类中的隐藏变量是否存在命名规范问题
                elif attr_n.endswith('_') and not attr_n.endswith('__'):
                    raise EnumOccupiedError(
                        f"ERROR: {attr_n} is should end with '__' but not '_', change a Name!!!")
                # 检查枚举类中的隐藏变量是否与父类变量存在命名冲突
                # 此处检查仅出于保险起见，如果此处报错，那就是enum库中SuperEnum方法的命名上出现了问题，因为命名空间应该已经完全划分开
                elif hasattr(cls_n.__base__, attr_n):
                    raise EnumOccupiedError(
                        f"ERROR: {attr_n} is conflicted with function in SuperEnum, change a Name!!!")
                # 检查
                else:
                    # 重置标志位
                    is_hide, is_sub = False, False
                    # 赋值枚举类型
                    val_n = getattr(cls_n, attr_n)
                    # 判断类型是否为子类
                    if isinstance(val_n, type) and issubclass(val_n, SuperEnum):
                        is_sub = True
                    # 递归并处理子类
                    if is_sub:
                        # 先递归
                        uuid_n = regress_enum(uuid_n, val_n)
                        # 赋值子级group属性
                        uuid2sub_n[uuid_n] = val_n
                        pack_n = _EnumPack(uuid=uuid_n, name=attr_n, value=val_n._value)
                        val_n._uuid, val_n._pack = uuid_n, pack_n
                    # 处理一般枚举成员
                    else:
                        pack_n = _EnumPack(uuid=uuid_n, name=attr_n, value=val_n)
                        setattr(cls_n, attr_n, uuid_n)
                    # 检查是否为隐藏枚举量并处理
                    if is_sub:
                        if val_n._hide:
                            is_hide = True
                    else:
                        if attr_n.endswith('__'):
                            is_hide = True
                    if is_hide:
                        uuid_hide_n.add(uuid_n)
                    # 如果为隐藏属性，则跳过配置index过程
                    if not is_hide:
                        # 赋值索引值，如果启用了顺序索引则填入对应位置，否则挂到最后
                        if is_seq:
                            try:
                                index = seq_n.index(attr_n)
                                index2uuid_n[index] = uuid_n
                            except ValueError:
                                raise EnumSequenceError(f"ERROR: '{attr_n}' is not in _sequence_!!!")
                        else:
                            index = index_n
                            index2uuid_n.append(uuid_n)
                        pack_n.index = index
                        # 刷新计数器
                        index_n += 1
                    else:
                        pack_n.index = None
                    uuid2pack_n[uuid_n] = pack_n
                    uuid_n += 1
            # 如果启用了顺序索引，则检查_sequence_是否全部被包含
            if is_seq:
                if index_n != seq_len:
                    raise EnumSequenceError(f"ERROR: index_n({index_n}) != _sequence_({seq_len})!!!")
            # 赋值本级group属性
            cls_n._index2uuid = index2uuid_n
            cls_n._length = index_n
            cls_n._uuid2pack = uuid2pack_n
            cls_n._uuid_hide = uuid_hide_n
            cls_n._uuid2sub = uuid2sub_n
            cls_n._uuid_all = index2uuid_n + list(uuid_hide_n)
            cls_n._uuid_sub = set(uuid2sub_n.keys())

            return uuid_n

        # 递归入口
        uuid = 0
        uuid = regress_enum(uuid, cls)
        # 赋值根目录本身的属性，本身一般仅用于占位，无实际意义
        cls._uuid = uuid
        cls._pack = _EnumPack(uuid=uuid, index=(None if cls._hide else 0), name=cls.__name__, value=cls._value)
        return cls

    return decorator
