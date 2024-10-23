#  `pdoc` 文档使用说明



编译指令，其中`latex_math=True`用于**启用数学输入功能**：

```shell
pdoc --output-dir docs hagike --html --config latex_math=True
```



pdoc支持**reST指令**，示例如下：

**注意**：

1. 缩进表示了reST指令的范围

2. `.. xxx::` 是第一级指令，`:xxx:` 是第二级指令

3. `.. image::` 指令的相对路径的根路径为对应python文件的文档所在的文件夹，图片是动态插入的，因此编译时若图片不存在不会报错，但显示时若图片不存在就会用后面的 `:alt:` 中的内容代替

4. `.. include::` 这里会拷贝进来文本文件中的指定区域，并用与直接写在docstring中相同的方式去进行解释。由于其是在编译时插入的，因此相对路径的根路径是源文件所在的文件夹，并且要求编译时必须存在，否则就会报错

5. 除了示例给出的admonitions(告诫，示例中的是`note`和`warning`)，类似的一共有这些：

   `attention`, `caution`, `danger`, `error`, `hint`, `important`, `note`, `tip`, `warning`, `admonition`

```python
def example_function_reST():
    """
    这是一个示例函数，用于展示如何在docstring中使用reST指令。
    .. note::
        这是一个注释，用于提供关于函数的额外信息。
    .. warning::
        使用此函数时请小心，因为它可能会执行危险的操作。
    .. image:: __res__/hagikehappy.jpg
        :alt: 这是一个图片的插入。请确保图片文件与文档在同一目录中，或者提供正确的路径。
    .. include:: __res__/included_content.txt
        :start-line: 2
        :end-line: 5
    .. math::
        E = mc^2
    这是一个数学表达式的示例，表示爱因斯坦的质能方程。
    .. versionadded:: 1.0
        此函数在版本1.0中添加。
    .. versionchanged:: 2.0
        在版本2.0中，此函数的性能得到了改进。
    .. deprecated:: 3.0
        请注意，此函数在版本3.0中已被弃用，建议使用`new_function`代替。
    .. todo::
        需要添加对此函数的更多测试。
    """
    pass
```



pdoc还支持各种**markdown语法**，包括（这里仅列出可用且符合预期的部分）：

- 图像插入，公式插入
- 斜体、加粗、下划线、框注、超链接
- 有序列表、无序列表

**注意**：pdoc处理的注释是经过转义的注释，因此markdown语法中的所有 `\` 本身都需要用 `\\`来进行转义，在公式中尤其如此

```python
def example_function_md():
    """
    这是一个示例函数，用于展示如何在docstring中使用markdown指令。 \n
    插入缩放过的图片： \n
    <img src="__res__/mine.jpg" style="zoom:80%;" /> \n
    插入公式： \n
    $$ \\alpha = \\beta + \\gamma $$ \n
    有序列表： \n
    1. Number One \n
    2. Number Two \n
    无序列表： \n
    - Level One \n
        - Level Two \n
            - Level Three \n
    """
    pass
```

此外，需要**注意**的还有：

1. pdoc中不会自动识别换行，因此换行需要手动插入转义字符`\n`
2. 此处约定**规范**：所有资源文件放在对应相对位置根目录下的`__res__/`文件夹中
3. 在两个 `\n` 换行之间，若所有行首部均有tab缩进，则相当于自动添加分块符 \`\`\`  xxx  \`\`\` 
4. 分块符内不需要再写 `\n` 了，换行会被自动检测
5. 在分块符内，非markdown写法是无效的，如Latex和reST
