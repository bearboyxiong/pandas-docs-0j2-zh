# 数据结构简介

> 原文：[Intro to Data Structures](http://pandas.pydata.org/pandas-docs/version/0.19.2/dsintro.html)

> 译者：[usyiyi.cn](http://usyiyi.cn/translate/Pandas_0j2/dsintro.html)

> 协议：[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/)

我们将首先快速，非全面地概述pandas中的基本数据结构，来让你起步。数据类型，索引和轴标记/对齐的基本行为适用于所有对象。为了起步，请导入numpy并将pandas加载到您的命名空间中：

```
In [1]: import numpy as np

In [2]: import pandas as pd

```

以下是一个基本原则：**数据对齐是内在的**。标签和数据之间的链接不会被破坏，除非你明确这样做。

我们将简要介绍数据结构，然后在单独的章节中，考虑所有功能和方法的大类。

## Series（序列）

[`Series`](generated/pandas.Series.html#pandas.Series "pandas.Series")是带有标签的一维数组，可以保存任何数据类型（整数，字符串，浮点数，Python对象等）。轴标签统称为**索引**。创建Series的基本方法是调用：

```
>>> s = pd.Series(data, index=index)

```

这里，`data`可以是许多不同的东西：

> *   Python dict（字典）
> *   ndarray
> *   标量值（如5）

传入的**索引**是轴标签的列表。因此，根据**数据的类型**，分为以下几种情况：

**来自ndarray**

如果`data`是ndarray，则**索引**必须与**数据**长度相同。如果没有传递索引，将创建值为`[0， ...， len(data) - 1]`的索引。

```
In [3]: s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

In [4]: s
Out[4]: 
a    0.2735
b    0.6052
c   -0.1692
d    1.8298
e    0.5432
dtype: float64

In [5]: s.index
Out[5]: Index([u'a', u'b', u'c', u'd', u'e'], dtype='object')

In [6]: pd.Series(np.random.randn(5))
Out[6]: 
0    0.3674
1   -0.8230
2   -1.0295
3   -1.0523
4   -0.8502
dtype: float64

```

注意

从v0.8.0开始，pandas支持非唯一索引值。如果尝试执行不支持重复索引值的操作，那么将会引发异常。延迟的原因几乎都基于性能（在计算中有很多实例，例如 GroupBy 的部分不使用索引）。

**来自字典**

如果`data`是字典，那么如果传入了**index**，则会取出数据中的值，对应于索引中的标签。否则，如果可能，将从字典的有序键构造索引。

```
In [7]: d = {'a' : 0., 'b' : 1., 'c' : 2.}

In [8]: pd.Series(d)
Out[8]: 
a    0.0
b    1.0
c    2.0
dtype: float64

In [9]: pd.Series(d, index=['b', 'c', 'd', 'a'])
Out[9]: 
b    1.0
c    2.0
d    NaN
a    0.0
dtype: float64

```

注意

NaN（不是数字）是用于pandas的标准缺失数据标记

**从标量值**：如果`data`是标量值，则必须提供索引。该值会重复，来匹配**索引**的长度。

```
In [10]: pd.Series(5., index=['a', 'b', 'c', 'd', 'e'])
Out[10]: 
a    5.0
b    5.0
c    5.0
d    5.0
e    5.0
dtype: float64

```

### Series 是类似于 ndarray 的[](#series-is-ndarray-like "Permalink to this headline")

`Series`的作用与`ndarray`非常相似，是大多数NumPy函数的有效参数。然而，像切片这样的东西也会对索引切片。

```
In [11]: s[0]
Out[11]: 0.27348116325673794

In [12]: s[:3]
Out[12]: 
a    0.2735
b    0.6052
c   -0.1692
dtype: float64

In [13]: s[s > s.median()]
Out[13]: 
b    0.6052
d    1.8298
dtype: float64

In [14]: s[[4, 3, 1]]
Out[14]: 
e    0.5432
d    1.8298
b    0.6052
dtype: float64

In [15]: np.exp(s)
Out[15]: 
a    1.3145
b    1.8317
c    0.8443
d    6.2327
e    1.7215
dtype: float64

```

我们将在单独的[章节](indexing.html#indexing)中强调基于数组的索引。

### Series 类似于字典[](#series-is-dict-like "Permalink to this headline")

Series就像一个固定大小的字典，您可以通过使用标签作为索引来获取和设置值：

```
In [16]: s['a']
Out[16]: 0.27348116325673794

In [17]: s['e'] = 12.

In [18]: s
Out[18]: 
a     0.2735
b     0.6052
c    -0.1692
d     1.8298
e    12.0000
dtype: float64

In [19]: 'e' in s
Out[19]: True

In [20]: 'f' in s
Out[20]: False

```

如果标签不存在，则会出现异常：

```
>>> s['f']
KeyError: 'f'

```

使用`get`方法，缺失的标签将返回None或指定的默认值：

```
In [21]: s.get('f')

In [22]: s.get('f', np.nan)
Out[22]: nan

```

另请参阅[属性访问](indexing.html#indexing-attribute-access)部分。

### Series 的向量化操作和标签对齐

进行数据分析时，像原始NumPy数组一样，一个值一个值地循环遍历序列通常不是必需的。Series 也可以传递给大多数期望 ndarray 的 NumPy 方法。

```
In [23]: s + s
Out[23]: 
a     0.5470
b     1.2104
c    -0.3385
d     3.6596
e    24.0000
dtype: float64

In [24]: s * 2
Out[24]: 
a     0.5470
b     1.2104
c    -0.3385
d     3.6596
e    24.0000
dtype: float64

In [25]: np.exp(s)
Out[25]: 
a         1.3145
b         1.8317
c         0.8443
d         6.2327
e    162754.7914
dtype: float64

```

Series 和 ndarray 之间的主要区别是，Series 上的操作会根据标签自动对齐数据。因此，您可以编写计算，而不考虑所涉及的 Series 是否具有相同标签。

```
In [26]: s[1:] + s[:-1]
Out[26]: 
a       NaN
b    1.2104
c   -0.3385
d    3.6596
e       NaN
dtype: float64

```

未对齐的 Series 之间的运算结果，将具有所涉及的索引的**并集**。如果在一个 Series 或其他系列中找不到某个标签，则结果将标记为`NaN`（缺失）。编写代码而不进行任何显式的数据对齐的能力，在交互式数据分析和研究中提供了巨大的自由和灵活性。pandas数据结构所集成的数据对齐特性，将pandas与用于处理标记数据的大多数相关工具分开。

注意

一般来说，我们选择使索引不同的对象之间的操作的默认结果为**union**，来避免信息的丢失。尽管缺少数据，拥有索引标签通常是重要信息，作为计算的一部分。您当然可以通过**dropna**函数，选择丢弃带有缺失数据的标签。

### 名称属性

Series还可以具有`name`属性：

```
In [27]: s = pd.Series(np.random.randn(5), name='something')

In [28]: s
Out[28]: 
0    1.5140
1   -1.2345
2    0.5666
3   -1.0184
4    0.1081
Name: something, dtype: float64

In [29]: s.name
Out[29]: 'something'

```

在多数情况下，Series 的`name`会自动赋值，特别是获取 DataFrame 的一维切片时，您将在下面看到它。

版本0.18.0中的新功能。

您可以使用[`pandas.Series.rename()`](generated/pandas.Series.rename.html#pandas.Series.rename "pandas.Series.rename")方法来重命名 Series。

```
In [30]: s2 = s.rename("different")

In [31]: s2.name
Out[31]: 'different'

```

注意，`s`和`s2`指向不同的对象。

## DataFrame（数据帧）

**DataFrame**是带有标签的二维数据结构，列的类型可能不同。你可以把它想象成一个电子表格或SQL表，或者 Series 对象的字典。它一般是最常用的pandas对象。像 Series 一样，DataFrame 接受许多不同类型的输入：

> *   一维数组，列表，字典或 Series 的字典
> *   二维 numpy.ndarray
> *   [结构化或记录](http://docs.scipy.org/doc/numpy/user/basics.rec.html) ndarray
> *   `Series`
> *   另一个`DataFrame`

和数据一起，您可以选择传递**index**（行标签）和**columns**（列标签）参数。如果传递索引或列，则会用于生成的DataFrame的索引或列。因此，Series 的字典加上特定索引将丢弃所有不匹配传入索引的数据。

如果轴标签未通过，则它们将基于常识规则从输入数据构造。

### 来自 Series 或字典的字典

结果的**index**是各种系列索引的**并集**。如果有任何嵌套的词典，这些将首先转换为Series。如果列没有传递，这些列将是字典的键的有序列表。

```
In [32]: d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
 ....:      'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
 ....: 

In [33]: df = pd.DataFrame(d)

In [34]: df
Out[34]: 
 one  two
a  1.0  1.0
b  2.0  2.0
c  3.0  3.0
d  NaN  4.0

In [35]: pd.DataFrame(d, index=['d', 'b', 'a'])
Out[35]: 
 one  two
d  NaN  4.0
b  2.0  2.0
a  1.0  1.0

In [36]: pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three'])
Out[36]: 
 two three
d  4.0   NaN
b  2.0   NaN
a  1.0   NaN

```

通过访问**index**和**column**属性可以分别访问行和列标签：

注意

同时传入一组特定的列和数据的字典时，传入的列将覆盖字典中的键。

```
In [37]: df.index
Out[37]: Index([u'a', u'b', u'c', u'd'], dtype='object')

In [38]: df.columns
Out[38]: Index([u'one', u'two'], dtype='object')

```

### 来自 ndarrays / lists 的字典

ndarrays 必须长度相同。如果传入了索引，它必须也与数组长度相同。如果没有传入索引，结果将是`range(n)`，其中`n`是数组长度。

```
In [39]: d = {'one' : [1., 2., 3., 4.],
 ....:      'two' : [4., 3., 2., 1.]}
 ....: 

In [40]: pd.DataFrame(d)
Out[40]: 
 one  two
0  1.0  4.0
1  2.0  3.0
2  3.0  2.0
3  4.0  1.0

In [41]: pd.DataFrame(d, index=['a', 'b', 'c', 'd'])
Out[41]: 
 one  two
a  1.0  4.0
b  2.0  3.0
c  3.0  2.0
d  4.0  1.0

```

### 来自结构化或记录数组

这种情况与数组的字典相同。

```
In [42]: data = np.zeros((2,), dtype=[('A', 'i4'),('B', 'f4'),('C', 'a10')])

In [43]: data[:] = [(1,2.,'Hello'), (2,3.,"World")]

In [44]: pd.DataFrame(data)
Out[44]: 
 A    B      C
0  1  2.0  Hello
1  2  3.0  World

In [45]: pd.DataFrame(data, index=['first', 'second'])
Out[45]: 
 A    B      C
first   1  2.0  Hello
second  2  3.0  World

In [46]: pd.DataFrame(data, columns=['C', 'A', 'B'])
Out[46]: 
 C  A    B
0  Hello  1  2.0
1  World  2  3.0

```

注意

DataFrame并不打算完全类似二维NumPy ndarray一样。

### 来自字典的数组

```
In [47]: data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

In [48]: pd.DataFrame(data2)
Out[48]: 
 a   b     c
0  1   2   NaN
1  5  10  20.0

In [49]: pd.DataFrame(data2, index=['first', 'second'])
Out[49]: 
 a   b     c
first   1   2   NaN
second  5  10  20.0

In [50]: pd.DataFrame(data2, columns=['a', 'b'])
Out[50]: 
 a   b
0  1   2
1  5  10

```

### 来自元组的字典

您可以通过传递元组字典来自动创建多索引的 DataFrame

```
In [51]: pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
 ....:               ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
 ....:               ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
 ....:               ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
 ....:               ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})
 ....: 
Out[51]: 
 a              b 
 a    b    c    a     b
A B  4.0  1.0  5.0  8.0  10.0
 C  3.0  2.0  6.0  7.0   NaN
 D  NaN  NaN  NaN  NaN   9.0

```

### 来自单个 Series

结果是一个 DataFrame，索引与输入的 Series 相同，并且单个列的名称是 Series 的原始名称（仅当没有提供其他列名时）。

**缺失数据**

在[缺失数据](missing_data.html#missing-data)部分中，将对此主题进行更多说明。为了构造具有缺失数据的DataFrame，请将`np.nan`用于缺失值。或者，您可以将`numpy.MaskedArray`作为数据参数传递给DataFrame构造函数，它屏蔽的条目将视为缺失值。

### 备选构造函数

**DataFrame.from_dict**

`DataFrame.from_dict`接受字典的字典或类似数组的序列的字典，并返回DataFrame。它的操作类似`DataFrame`的构造函数，除了默认情况下为`'columns'`的`orient`参数，但它可以设置为`'index'`，以便将字典的键用作行标签。

**DataFrame.from_records**

`DataFrame.from_records`首届元组的列表或带有结构化dtype的ndarray。它的工作方式类似于正常`DataFrame`构造函数，除了索引可能是结构化dtype的特定字段。例如：

```
In [52]: data
Out[52]: 
array([(1, 2.0, 'Hello'), (2, 3.0, 'World')], 
 dtype=[('A', '<i4'), ('B', '<f4'), ('C', 'S10')])

In [53]: pd.DataFrame.from_records(data, index='C')
Out[53]: 
 A    B
C 
Hello  1  2.0
World  2  3.0

```

**DataFrame.from_items**

`DataFrame.from_items`类似于`字典`的构造函数，它接受`键 值`对的序列，其中的键是列标签（或在`orient ='index'`的情况下是行标签），值是列的值（或行的值）。对于构建列为特定的顺序的DataFrame，而不必传递明确的列的列表，它非常有用：

```
In [54]: pd.DataFrame.from_items([('A', [1, 2, 3]), ('B', [4, 5, 6])])
Out[54]: 
 A  B
0  1  4
1  2  5
2  3  6

```

如果您传入`orient='index'`，键将是行标签。但在这种情况下，您还必须传递所需的列名称：

```
In [55]: pd.DataFrame.from_items([('A', [1, 2, 3]), ('B', [4, 5, 6])],
 ....:                         orient='index', columns=['one', 'two', 'three'])
 ....: 
Out[55]: 
 one  two  three
A    1    2      3
B    4    5      6

```

### 列的选取、添加、删除

你可以在语义上，将 DataFrame 当做 Series 对象的字典来处理。列的获取，设置和删除的方式与字典操作的语法相同：

```
In [56]: df['one']
Out[56]: 
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64

In [57]: df['three'] = df['one'] * df['two']

In [58]: df['flag'] = df['one'] > 2

In [59]: df
Out[59]: 
 one  two  three   flag
a  1.0  1.0    1.0  False
b  2.0  2.0    4.0  False
c  3.0  3.0    9.0   True
d  NaN  4.0    NaN  False

```

列可以像字典一样删除或弹出：

```
In [60]: del df['two']

In [61]: three = df.pop('three')

In [62]: df
Out[62]: 
 one   flag
a  1.0  False
b  2.0  False
c  3.0   True
d  NaN  False

```

当插入一个标量值时，它自然会广播来填充该列：

```
In [63]: df['foo'] = 'bar'

In [64]: df
Out[64]: 
 one   flag  foo
a  1.0  False  bar
b  2.0  False  bar
c  3.0   True  bar
d  NaN  False  bar

```

当插入的 Series 与 DataFrame 的索引不同时，它将适配 DataFrame 的索引：

```
In [65]: df['one_trunc'] = df['one'][:2]

In [66]: df
Out[66]: 
 one   flag  foo  one_trunc
a  1.0  False  bar        1.0
b  2.0  False  bar        2.0
c  3.0   True  bar        NaN
d  NaN  False  bar        NaN

```

您可以插入原始的ndarray，但它们的长度必须匹配DataFrame的索引的长度。

默认情况下，列在末尾插入。`insert`函数可用于在列中的特定位置插入：

```
In [67]: df.insert(1, 'bar', df['one'])

In [68]: df
Out[68]: 
 one  bar   flag  foo  one_trunc
a  1.0  1.0  False  bar        1.0
b  2.0  2.0  False  bar        2.0
c  3.0  3.0   True  bar        NaN
d  NaN  NaN  False  bar        NaN

```

### 使用方法链来创建新的列

版本0.16.0中的新功能。

受[dplyr](http://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html#mutate)的`mutate`动词的启发，DataFrame 拥有[`assign()`](generated/pandas.DataFrame.assign.html#pandas.DataFrame.assign "pandas.DataFrame.assign")方法，允许您轻易创建新的列，它可能从现有列派生。

```
In [69]: iris = pd.read_csv('data/iris.data')

In [70]: iris.head()
Out[70]: 
 SepalLength  SepalWidth  PetalLength  PetalWidth         Name
0          5.1         3.5          1.4         0.2  Iris-setosa
1          4.9         3.0          1.4         0.2  Iris-setosa
2          4.7         3.2          1.3         0.2  Iris-setosa
3          4.6         3.1          1.5         0.2  Iris-setosa
4          5.0         3.6          1.4         0.2  Iris-setosa

In [71]: (iris.assign(sepal_ratio = iris['SepalWidth'] / iris['SepalLength'])
 ....:      .head())
 ....: 
Out[71]: 
 SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa       0.6863
1          4.9         3.0          1.4         0.2  Iris-setosa       0.6122
2          4.7         3.2          1.3         0.2  Iris-setosa       0.6809
3          4.6         3.1          1.5         0.2  Iris-setosa       0.6739
4          5.0         3.6          1.4         0.2  Iris-setosa       0.7200

```

上面是插入预计算值的示例。我们还可以传递函数作为参数，这个函数会在 DataFrame 上调用，结果会添加给 DataFrame。

```
In [72]: iris.assign(sepal_ratio = lambda x: (x['SepalWidth'] /
 ....:                                      x['SepalLength'])).head()
 ....: 
Out[72]: 
 SepalLength  SepalWidth  PetalLength  PetalWidth         Name  sepal_ratio
0          5.1         3.5          1.4         0.2  Iris-setosa       0.6863
1          4.9         3.0          1.4         0.2  Iris-setosa       0.6122
2          4.7         3.2          1.3         0.2  Iris-setosa       0.6809
3          4.6         3.1          1.5         0.2  Iris-setosa       0.6739
4          5.0         3.6          1.4         0.2  Iris-setosa       0.7200

```

`assign` **始终**返回数据的副本，而保留原始DataFrame不变。

传递可调用对象，而不是要插入的实际值，当您没有现有 DataFrame 的引用时，它很有用。在操作链中使用`assign`时，这很常见。

```
In [73]: (iris.query('SepalLength > 5')
 ....:      .assign(SepalRatio = lambda x: x.SepalWidth / x.SepalLength,
 ....:              PetalRatio = lambda x: x.PetalWidth / x.PetalLength)
 ....:      .plot(kind='scatter', x='SepalRatio', y='PetalRatio'))
 ....: 
Out[73]: <matplotlib.axes._subplots.AxesSubplot at 0x7ff286891b50>

```

![http://pandas.pydata.org/pandas-docs/version/0.19.2/_images/basics_assign.png](http://pandas.pydata.org/pandas-docs/version/0.19.2/_images/basics_assign.png)

由于传入了一个函数，因此该函数在 DataFrame 上求值。重要的是，这个 DataFrame 已经过滤为 sepal 长度大于 5 的那些行。首先进行过滤，然后计算比值。这是一个示例，其中我们没有_被过滤的_ DataFrame的可用引用。

`assign`函数的参数是`**kwargs`。键是新字段的列名称，值是要插入的值（例如，`Series`或NumPy数组），或者是个函数，它在`DataFrame`上调用。返回原始DataFrame的_副本_，它插入了新值。

警告

由于`assign`的函数签名为`**kwargs`，因此不能保证在产生的DataFrame中，新列的顺序与传递的顺序一致。为了使事情可预测，条目按字典序（按键）插入到 DataFrame 的末尾。

首先计算所有表达式，然后赋值。因此，在`assign`的同一调用中，您不能引用要赋值的另一列。例如：

> ```
> In [74]: # Don't do this, bad reference to `C`
>  df.assign(C = lambda x: x['A'] + x['B'],
>  D = lambda x: x['A'] + x['C'])
> In [2]: # Instead, break it into two assigns
>  (df.assign(C = lambda x: x['A'] + x['B'])
>  .assign(D = lambda x: x['A'] + x['C']))
> 
> ```

### 索引 / 选取

索引的基本方式如下：

<colgroup><col width="50%"> <col width="33%"> <col width="17%"></colgroup> 
| 操作 | 句法 | 结果 |
| --- | --- | --- |
| 选择列 | `df[col]` | Series |
| 按标签选择行 | `df.loc[label]` | Series |
| 按整数位置选择行 | `df.iloc[loc]` | Series |
| 对行切片 | `df[5:10]` | DataFrame |
| 通过布尔向量选择行 | `df[bool_vec]` | DataFrame |

例如，行的选择返回 Series，其索引是 DataFrame 的列：

```
In [75]: df.loc['b']
Out[75]: 
one              2
bar              2
flag         False
foo            bar
one_trunc        2
Name: b, dtype: object

In [76]: df.iloc[2]
Out[76]: 
one             3
bar             3
flag         True
foo           bar
one_trunc     NaN
Name: c, dtype: object

```

对于更复杂的基于标签的索引和切片的更详尽的处理，请参阅[索引章节](indexing.html#indexing)。我们将在[重索引章节](basics.html#basics-reindexing)中，强调重索引/适配新标签集的基本原理。

### 数据对齐和算术

DataFrame对象之间的数据自动按照**列和索引（行标签）**对齐。同样，生成的对象具有列和行标签的并集。

```
In [77]: df = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])

In [78]: df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])

In [79]: df + df2
Out[79]: 
 A       B       C   D
0  0.5222  0.3225 -0.7566 NaN
1 -0.8441  0.2334  0.8818 NaN
2 -2.2079 -0.1572 -0.3875 NaN
3  2.8080 -1.0927  1.0432 NaN
4 -1.7511 -2.0812  2.7477 NaN
5 -3.2473 -1.0850  0.7898 NaN
6 -1.7107  0.0661  0.1294 NaN
7     NaN     NaN     NaN NaN
8     NaN     NaN     NaN NaN
9     NaN     NaN     NaN NaN

```

执行 DataFrame和Series之间的操作时，默认行为是，将Dataframe 的**列****索引**与 Series 对齐，从而按行[广播](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)。例如：

```
In [80]: df - df.iloc[0]
Out[80]: 
 A       B       C       D
0  0.0000  0.0000  0.0000  0.0000
1 -2.6396 -1.0702  1.7214 -0.7896
2 -2.7662 -1.6918  2.2776 -2.5401
3  0.8679 -3.5247  1.9365 -0.1331
4 -1.9883 -3.2162  2.0464 -1.0700
5 -3.3932 -4.0976  1.6366 -2.1635
6 -1.3668 -1.9572  1.6523 -0.7191
7 -0.7949 -2.1663  0.9706 -2.6297
8 -0.8383 -1.3630  1.6702 -2.0865
9  0.8588  0.0814  3.7305 -1.3737

```

在处理时间序列数据的特殊情况下，DataFrame索引也包含日期，广播是按列的方式：

```
In [81]: index = pd.date_range('1/1/2000', periods=8)

In [82]: df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))

In [83]: df
Out[83]: 
 A       B       C
2000-01-01  0.2731  0.3604 -1.1515
2000-01-02  1.1577  1.4787 -0.6528
2000-01-03 -0.7712  0.2203 -0.5739
2000-01-04 -0.6356 -1.1703 -0.0789
2000-01-05 -1.4687  0.1705 -1.8796
2000-01-06 -1.2037  0.9568 -1.1383
2000-01-07 -0.6540 -0.2169  0.3843
2000-01-08 -2.1639 -0.8145 -1.2475

In [84]: type(df['A'])
Out[84]: pandas.core.series.Series

In [85]: df - df['A']
Out[85]: 
 2000-01-01 00:00:00  2000-01-02 00:00:00  2000-01-03 00:00:00  \
2000-01-01                  NaN                  NaN                  NaN 
2000-01-02                  NaN                  NaN                  NaN 
2000-01-03                  NaN                  NaN                  NaN 
2000-01-04                  NaN                  NaN                  NaN 
2000-01-05                  NaN                  NaN                  NaN 
2000-01-06                  NaN                  NaN                  NaN 
2000-01-07                  NaN                  NaN                  NaN 
2000-01-08                  NaN                  NaN                  NaN 

 2000-01-04 00:00:00 ...  2000-01-08 00:00:00   A   B   C 
2000-01-01                  NaN ...                  NaN NaN NaN NaN 
2000-01-02                  NaN ...                  NaN NaN NaN NaN 
2000-01-03                  NaN ...                  NaN NaN NaN NaN 
2000-01-04                  NaN ...                  NaN NaN NaN NaN 
2000-01-05                  NaN ...                  NaN NaN NaN NaN 
2000-01-06                  NaN ...                  NaN NaN NaN NaN 
2000-01-07                  NaN ...                  NaN NaN NaN NaN 
2000-01-08                  NaN ...                  NaN NaN NaN NaN 

[8 rows x 11 columns]

```

警告

```
df - df['A']

```

现已弃用，将在以后的版本中删除。复现此行为的首选方法是

```
df.sub(df['A'], axis=0)

```

对于显式控制匹配和广播行为，请参阅[灵活的二元运算](basics.html#basics-binop)一节。

标量的操作正如你的预期：

```
In [86]: df * 5 + 2
Out[86]: 
 A       B       C
2000-01-01  3.3655  3.8018 -3.7575
2000-01-02  7.7885  9.3936 -1.2641
2000-01-03 -1.8558  3.1017 -0.8696
2000-01-04 -1.1781 -3.8513  1.6056
2000-01-05 -5.3437  2.8523 -7.3982
2000-01-06 -4.0186  6.7842 -3.6915
2000-01-07 -1.2699  0.9157  3.9217
2000-01-08 -8.8194 -2.0724 -4.2375

In [87]: 1 / df
Out[87]: 
 A       B        C
2000-01-01  3.6616  2.7751  -0.8684
2000-01-02  0.8638  0.6763  -1.5318
2000-01-03 -1.2967  4.5383  -1.7424
2000-01-04 -1.5733 -0.8545 -12.6759
2000-01-05 -0.6809  5.8662  -0.5320
2000-01-06 -0.8308  1.0451  -0.8785
2000-01-07 -1.5291 -4.6113   2.6019
2000-01-08 -0.4621 -1.2278  -0.8016

In [88]: df ** 4
Out[88]: 
 A       B           C
2000-01-01   0.0056  0.0169  1.7581e+00
2000-01-02   1.7964  4.7813  1.8162e-01
2000-01-03   0.3537  0.0024  1.0849e-01
2000-01-04   0.1632  1.8755  3.8733e-05
2000-01-05   4.6534  0.0008  1.2482e+01
2000-01-06   2.0995  0.8382  1.6789e+00
2000-01-07   0.1829  0.0022  2.1819e-02
2000-01-08  21.9244  0.4401  2.4219e+00

```

布尔运算符也同样有效：

```
In [89]: df1 = pd.DataFrame({'a' : [1, 0, 1], 'b' : [0, 1, 1] }, dtype=bool)

In [90]: df2 = pd.DataFrame({'a' : [0, 1, 1], 'b' : [1, 1, 0] }, dtype=bool)

In [91]: df1 & df2
Out[91]: 
 a      b
0  False  False
1  False   True
2   True  False

In [92]: df1 | df2
Out[92]: 
 a     b
0  True  True
1  True  True
2  True  True

In [93]: df1 ^ df2
Out[93]: 
 a      b
0   True   True
1   True  False
2  False   True

In [94]: -df1
Out[94]: 
 a      b
0  False   True
1   True  False
2  False  False

```

### 转置

对于转置，访问`T`属性（`transpose`函数也是），类似于ndarray：

```
# only show the first 5 rows
In [95]: df[:5].T
Out[95]: 
 2000-01-01  2000-01-02  2000-01-03  2000-01-04  2000-01-05
A      0.2731      1.1577     -0.7712     -0.6356     -1.4687
B      0.3604      1.4787      0.2203     -1.1703      0.1705
C     -1.1515     -0.6528     -0.5739     -0.0789     -1.8796

```

### DataFrame 与 NumPy 函数的互操作

逐元素的 NumPy ufunc（log，exp，sqrt，...）和各种其他NumPy函数可以无缝用于DataFrame，假设其中的数据是数字：

```
In [96]: np.exp(df)
Out[96]: 
 A       B       C
2000-01-01  1.3140  1.4338  0.3162
2000-01-02  3.1826  4.3873  0.5206
2000-01-03  0.4625  1.2465  0.5633
2000-01-04  0.5296  0.3103  0.9241
2000-01-05  0.2302  1.1859  0.1526
2000-01-06  0.3001  2.6034  0.3204
2000-01-07  0.5200  0.8050  1.4686
2000-01-08  0.1149  0.4429  0.2872

In [97]: np.asarray(df)
Out[97]: 
array([[ 0.2731,  0.3604, -1.1515],
 [ 1.1577,  1.4787, -0.6528],
 [-0.7712,  0.2203, -0.5739],
 [-0.6356, -1.1703, -0.0789],
 [-1.4687,  0.1705, -1.8796],
 [-1.2037,  0.9568, -1.1383],
 [-0.654 , -0.2169,  0.3843],
 [-2.1639, -0.8145, -1.2475]])

```

DataFrame上的dot方法实现了矩阵乘法：

```
In [98]: df.T.dot(df)
Out[98]: 
 A       B       C
A  11.1298  2.8864  6.0015
B   2.8864  5.3895 -1.8913
C   6.0015 -1.8913  8.6204

```

类似地，Series上的dot方法实现了点积：

```
In [99]: s1 = pd.Series(np.arange(5,10))

In [100]: s1.dot(s1)
Out[100]: 255

```

DataFrame不打算作为ndarray的替代品，因为它的索引语义和矩阵是非常不同的。

### 控制台展示

非常大的DataFrames将被截断，来在控制台中展示。您也可以使用[`info()`](generated/pandas.DataFrame.info.html#pandas.DataFrame.info "pandas.DataFrame.info")取得摘要。（这里我从**plyr** R软件包中，读取CSV版本的**棒球**数据集）：

```
In [101]: baseball = pd.read_csv('data/baseball.csv')

In [102]: print(baseball)
 id     player  year  stint  ...   hbp   sh   sf  gidp
0   88641  womacto01  2006      2  ...   0.0  3.0  0.0   0.0
1   88643  schilcu01  2006      1  ...   0.0  0.0  0.0   0.0
..    ...        ...   ...    ...  ...   ...  ...  ...   ...
98  89533   aloumo01  2007      1  ...   2.0  0.0  3.0  13.0
99  89534  alomasa02  2007      1  ...   0.0  0.0  0.0   0.0

[100 rows x 23 columns]

In [103]: baseball.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100 entries, 0 to 99
Data columns (total 23 columns):
id        100 non-null int64
player    100 non-null object
year      100 non-null int64
stint     100 non-null int64
team      100 non-null object
lg        100 non-null object
g         100 non-null int64
ab        100 non-null int64
r         100 non-null int64
h         100 non-null int64
X2b       100 non-null int64
X3b       100 non-null int64
hr        100 non-null int64
rbi       100 non-null float64
sb        100 non-null float64
cs        100 non-null float64
bb        100 non-null int64
so        100 non-null float64
ibb       100 non-null float64
hbp       100 non-null float64
sh        100 non-null float64
sf        100 non-null float64
gidp      100 non-null float64
dtypes: float64(9), int64(11), object(3)
memory usage: 18.0+ KB

```

但是，使用`to_string`将返回表格形式的DataFrame的字符串表示，但并不总是适合控制台宽度：

```
In [104]: print(baseball.iloc[-20:, :12].to_string())
 id     player  year  stint team  lg    g   ab   r    h  X2b  X3b
80  89474  finlest01  2007      1  COL  NL   43   94   9   17    3    0
81  89480  embreal01  2007      1  OAK  AL    4    0   0    0    0    0
82  89481  edmonji01  2007      1  SLN  NL  117  365  39   92   15    2
83  89482  easleda01  2007      1  NYN  NL   76  193  24   54    6    0
84  89489  delgaca01  2007      1  NYN  NL  139  538  71  139   30    0
85  89493  cormirh01  2007      1  CIN  NL    6    0   0    0    0    0
86  89494  coninje01  2007      2  NYN  NL   21   41   2    8    2    0
87  89495  coninje01  2007      1  CIN  NL   80  215  23   57   11    1
88  89497  clemero02  2007      1  NYA  AL    2    2   0    1    0    0
89  89498  claytro01  2007      2  BOS  AL    8    6   1    0    0    0
90  89499  claytro01  2007      1  TOR  AL   69  189  23   48   14    0
91  89501  cirilje01  2007      2  ARI  NL   28   40   6    8    4    0
92  89502  cirilje01  2007      1  MIN  AL   50  153  18   40    9    2
93  89521  bondsba01  2007      1  SFN  NL  126  340  75   94   14    0
94  89523  biggicr01  2007      1  HOU  NL  141  517  68  130   31    3
95  89525  benitar01  2007      2  FLO  NL   34    0   0    0    0    0
96  89526  benitar01  2007      1  SFN  NL   19    0   0    0    0    0
97  89530  ausmubr01  2007      1  HOU  NL  117  349  38   82   16    3
98  89533   aloumo01  2007      1  NYN  NL   87  328  51  112   19    1
99  89534  alomasa02  2007      1  NYN  NL    8   22   1    3    1    0

```

从0.10.0版本开始，默认情况下，宽的 DataFrames 以多行打印：

```
In [105]: pd.DataFrame(np.random.randn(3, 12))
Out[105]: 
 0         1         2         3         4         5         6   \
0  2.173014  1.273573  0.888325  0.631774  0.206584 -1.745845 -0.505310 
1 -1.240418  2.177280 -0.082206  0.827373 -0.700792  0.524540 -1.101396 
2  0.269598 -0.453050 -1.821539 -0.126332 -0.153257  0.405483 -0.504557 

 7         8         9         10        11 
0  1.376623  0.741168 -0.509153 -2.012112 -1.204418 
1  1.115750  0.294139  0.286939  1.709761 -0.212596 
2  1.405148  0.778061 -0.799024 -0.670727  0.086877 

```

您可以通过设置`display.width`选项，更改单行上的打印量：

```
In [106]: pd.set_option('display.width', 40) # default is 80

In [107]: pd.DataFrame(np.random.randn(3, 12))
Out[107]: 
 0         1         2   \
0  1.179465  0.777427 -1.923460 
1  0.054928  0.776156  0.372060 
2 -0.243404 -1.506557 -1.977226 

 3         4         5   \
0  0.782432  0.203446  0.250652 
1  0.710963 -0.784859  0.168405 
2 -0.226582 -0.777971  0.231309 

 6         7         8   \
0 -2.349580 -0.540814 -0.748939 
1  0.159230  0.866492  1.266025 
2  1.394479  0.723474 -0.097256 

 9         10        11 
0 -0.994345  1.478624 -0.341991 
1  0.555240  0.731803  0.219383 
2  0.375274 -0.314401 -2.363136 

```

您可以通过设置`display.max_colwidth`来调整各列的最大宽度

```
In [108]: datafile={'filename': ['filename_01','filename_02'],
 .....:           'path': ["media/user_name/storage/folder_01/filename_01",
 .....:                    "media/user_name/storage/folder_02/filename_02"]}
 .....: 

In [109]: pd.set_option('display.max_colwidth',30)

In [110]: pd.DataFrame(datafile)
Out[110]: 
 filename  \
0  filename_01 
1  filename_02 

 path 
0  media/user_name/storage/fo... 
1  media/user_name/storage/fo... 

In [111]: pd.set_option('display.max_colwidth',100)

In [112]: pd.DataFrame(datafile)
Out[112]: 
 filename  \
0  filename_01 
1  filename_02 

 path 
0  media/user_name/storage/folder_01/filename_01 
1  media/user_name/storage/folder_02/filename_02 

```

您也可以通过`expand_frame_repr`选项停用此功能。这将表打印在一个块中。

### DataFrame 列属性访问和 IPython 补全

如果DataFrame列标签是有效的Python变量名，则可以像属性一样访问该列：

```
In [113]: df = pd.DataFrame({'foo1' : np.random.randn(5),
 .....:                    'foo2' : np.random.randn(5)})
 .....: 

In [114]: df
Out[114]: 
 foo1      foo2
0 -0.412237  0.213232
1 -0.237644  1.740139
2  1.272869 -0.241491
3  1.220450 -0.868514
4  1.315172  0.407544

In [115]: df.foo1
Out[115]: 
0   -0.412237
1   -0.237644
2    1.272869
3    1.220450
4    1.315172
Name: foo1, dtype: float64

```

这些列还连接了[IPython](http://ipython.org)补全机制，因此可以通过制表符补全：

```
In [5]: df.fo<TAB>
df.foo1  df.foo2

```

## Panel（面板）

Panel是一个稍微不常用的容器，但是对于三维数据仍然重要。术语[面板数据](http://en.wikipedia.org/wiki/Panel_data)源自计量经济学，是pandas名称的部分来源：pan(el)-da(ta)-s。 三个轴旨在提供一些语义上的含义，来描述涉及面板数据的操作，特别是面板数据的计量分析。但是，出于切割DataFrame对象的集合的严格目的，您可能会发现轴名称稍有任意：

> *   **items**（条目）：轴0，每个条目对应于其中包含的DataFrame
> *   **major_axis**（主轴）：轴1，它是每个DataFrame的**index**（行）
> *   **minor_axis**（副轴）：轴2，它是每个DataFrames的**columns**（列）

Panel 的构造正如你的期望：

### 来自三维 ndarray 和可选的轴标签

```
In [116]: wp = pd.Panel(np.random.randn(2, 5, 4), items=['Item1', 'Item2'],
 .....:               major_axis=pd.date_range('1/1/2000', periods=5),
 .....:               minor_axis=['A', 'B', 'C', 'D'])
 .....: 

In [117]: wp
Out[117]: 
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 5 (major_axis) x 4 (minor_axis)
Items axis: Item1 to Item2
Major_axis axis: 2000-01-01 00:00:00 to 2000-01-05 00:00:00
Minor_axis axis: A to D

```

### 来自 DataFrame 对象的字典

```
In [118]: data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
 .....:         'Item2' : pd.DataFrame(np.random.randn(4, 2))}
 .....: 

In [119]: pd.Panel(data)
Out[119]: 
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 4 (major_axis) x 3 (minor_axis)
Items axis: Item1 to Item2
Major_axis axis: 0 to 3
Minor_axis axis: 0 to 2

```

注意，字典中的值只需要**可转换为DataFrame**。因此，它们可以是DataFrame的任何其他有效输入，像上面一样。

一个有用的工厂方法是`Panel.from_dict`，它接受上面的DataFrames的字典，以及以下命名参数：

<colgroup><col width="17%"> <col width="17%"> <col width="67%"></colgroup> 
| 参数 | 默认 | 描述 |
| --- | --- | --- |
| intersect（交集） | `False` | 丢弃索引没有对齐的元素 |
| orient（方向） | `items` | 使用`minor`将DataFrames的列用作 Panel 的条目 |

例如，与上面的构造相比：

```
In [120]: pd.Panel.from_dict(data, orient='minor')
Out[120]: 
<class 'pandas.core.panel.Panel'>
Dimensions: 3 (items) x 4 (major_axis) x 2 (minor_axis)
Items axis: 0 to 2
Major_axis axis: 0 to 3
Minor_axis axis: Item1 to Item2

```

Orient对于混合类型的DataFrames特别有用。如果你传递一个DataFrame对象的字典，它的列是混合类型，所有的数据将转换为`dtype=object`，除非你传递`orient='minor'`：

```
In [121]: df = pd.DataFrame({'a': ['foo', 'bar', 'baz'],
 .....:                    'b': np.random.randn(3)})
 .....: 

In [122]: df
Out[122]: 
 a         b
0  foo -1.142863
1  bar -1.015321
2  baz  0.683625

In [123]: data = {'item1': df, 'item2': df}

In [124]: panel = pd.Panel.from_dict(data, orient='minor')

In [125]: panel['a']
Out[125]: 
 item1 item2
0   foo   foo
1   bar   bar
2   baz   baz

In [126]: panel['b']
Out[126]: 
 item1     item2
0 -1.142863 -1.142863
1 -1.015321 -1.015321
2  0.683625  0.683625

In [127]: panel['b'].dtypes
Out[127]: 
item1    float64
item2    float64
dtype: object

```

注意

不幸的是，面板比Series和DataFrame更不常用，在特性方面略有忽略。DataFrame中提供的许多方法和选项在Panel中不可用。这将会得到处理，当然，是未来的版本中。如果你加入我的代码库，会更快。

### 来自 DataFrame，使用`to_panel` 方法

此方法在v0.7中引入，来替换`LongPanel.to_long`，并将具有二级索引的DataFrame转换为Panel。

```
In [128]: midx = pd.MultiIndex(levels=[['one', 'two'], ['x','y']], labels=[[1,1,0,0],[1,0,1,0]])

In [129]: df = pd.DataFrame({'A' : [1, 2, 3, 4], 'B': [5, 6, 7, 8]}, index=midx)

In [130]: df.to_panel()
Out[130]: 
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 2 (major_axis) x 2 (minor_axis)
Items axis: A to B
Major_axis axis: one to two
Minor_axis axis: x to y

```

### 条目选取 / 添加 / 删除

类似于DataFrame作为 Series 的字典，Panel就像是DataFrames的字典：

```
In [131]: wp['Item1']
Out[131]: 
 A         B         C         D
2000-01-01 -0.729430  0.427693 -0.121325 -0.736418
2000-01-02  0.739037 -0.648805 -0.383057  0.385027
2000-01-03  2.321064 -1.290881  0.105458 -1.097035
2000-01-04  0.158759 -1.261191 -0.081710  1.390506
2000-01-05 -1.962031 -0.505580  0.021253 -0.317071

In [132]: wp['Item3'] = wp['Item1'] / wp['Item2']

```

用于插入和删除的API与DataFrame相同。和DataFrame一样，如果条目是一个有效的Python标识符，您可以作为一个属性访问它，并在IPython中补全它。

### 转置

可以使用 Panel 的`transpose`方法（除非数据是异构的，否则它不会默认制作副本）来重新排列它：

```
In [133]: wp.transpose(2, 0, 1)
Out[133]: 
<class 'pandas.core.panel.Panel'>
Dimensions: 4 (items) x 3 (major_axis) x 5 (minor_axis)
Items axis: A to D
Major_axis axis: Item1 to Item3
Minor_axis axis: 2000-01-01 00:00:00 to 2000-01-05 00:00:00

```

### 索引 / 选取

<colgroup><col width="50%"> <col width="33%"> <col width="17%"></colgroup> 
| 操作 | 句法 | 结果 |
| --- | --- | --- |
| 选取条目 | `wp[item]` | DataFrame |
| 选取主轴标签 | `wp.major_xs(val)` | DataFrame |
| 选取副轴标签 | `wp.minor_xs(val)` | DataFrame |

例如，使用之前的示例数据，我们可以执行：

```
In [134]: wp['Item1']
Out[134]: 
 A         B         C         D
2000-01-01 -0.729430  0.427693 -0.121325 -0.736418
2000-01-02  0.739037 -0.648805 -0.383057  0.385027
2000-01-03  2.321064 -1.290881  0.105458 -1.097035
2000-01-04  0.158759 -1.261191 -0.081710  1.390506
2000-01-05 -1.962031 -0.505580  0.021253 -0.317071

In [135]: wp.major_xs(wp.major_axis[2])
Out[135]: 
 Item1     Item2     Item3
A  2.321064 -0.538606 -4.309389
B -1.290881  0.791512 -1.630905
C  0.105458 -0.020302 -5.194337
D -1.097035  0.184430 -5.948253

In [136]: wp.minor_axis
Out[136]: Index([u'A', u'B', u'C', u'D'], dtype='object')

In [137]: wp.minor_xs('C')
Out[137]: 
 Item1     Item2     Item3
2000-01-01 -0.121325  1.413524 -0.085832
2000-01-02 -0.383057  1.243178 -0.308127
2000-01-03  0.105458 -0.020302 -5.194337
2000-01-04 -0.081710 -1.811565  0.045105
2000-01-05  0.021253 -1.040542 -0.020425

```

### 挤压

改变对象的维度的另一种方式是`squeeze`（挤压）长度为 1 的对象，类似于`wp['Item1']`

```
In [138]: wp.reindex(items=['Item1']).squeeze()
Out[138]: 
 A         B         C         D
2000-01-01 -0.729430  0.427693 -0.121325 -0.736418
2000-01-02  0.739037 -0.648805 -0.383057  0.385027
2000-01-03  2.321064 -1.290881  0.105458 -1.097035
2000-01-04  0.158759 -1.261191 -0.081710  1.390506
2000-01-05 -1.962031 -0.505580  0.021253 -0.317071

In [139]: wp.reindex(items=['Item1'], minor=['B']).squeeze()
Out[139]: 
2000-01-01    0.427693
2000-01-02   -0.648805
2000-01-03   -1.290881
2000-01-04   -1.261191
2000-01-05   -0.505580
Freq: D, Name: B, dtype: float64

```

### 转换为 DataFrame

Panel 可以以二维形式表示为层次索引的 DataFrame。详细信息，请参阅[层次索引](advanced.html#advanced-hierarchical)一节。为了将Panel转换为DataFrame，请使用`to_frame`方法：

```
In [140]: panel = pd.Panel(np.random.randn(3, 5, 4), items=['one', 'two', 'three'],
 .....:                  major_axis=pd.date_range('1/1/2000', periods=5),
 .....:                  minor_axis=['a', 'b', 'c', 'd'])
 .....: 

In [141]: panel.to_frame()
Out[141]: 
 one       two     three
major      minor 
2000-01-01 a     -1.876826 -0.383171 -0.117339
 b     -1.873827 -0.172217  0.780048
 c     -0.251457 -1.674685  2.162047
 d      0.027599  0.762474  0.874233
2000-01-02 a      1.235291  0.481666 -0.764147
 b      0.850574  1.217546 -0.484495
 c     -1.140302  0.577103  0.298570
 d      2.149143 -0.076021  0.825136
2000-01-03 a      0.504452  0.720235 -0.388020
 b      0.678026  0.202660 -0.339279
 c     -0.628443 -0.314950  0.141164
 d      1.191156 -0.410852  0.565930
2000-01-04 a     -1.145363  0.542758 -1.749969
 b     -0.523153  1.955407 -1.402941
 c     -1.299878 -0.940645  0.623222
 d     -0.110240  0.076257  0.020129
2000-01-05 a     -0.333712 -0.897159 -2.858463
 b      0.416876 -1.265679  0.885765
 c     -0.436400 -0.528311  0.158014
 d      0.999768 -0.660014 -1.981797

```

## Panel4D 和 PanelND (废弃)

警告

在0.19.0 中，`Panel4D`和`PanelND`已弃用，并且将在以后的版本中删除。表示这些类型的n维数据的推荐方法是使用[xarray软件包](http://xarray.pydata.org/en/stable/)。Pandas提供了一个[`to_xarray()`](generated/pandas.Panel4D.to_xarray.html#pandas.Panel4D.to_xarray "pandas.Panel4D.to_xarray")方法来自动执行此转换。

这些对象的文档，请参见以前版本的[文档](http://pandas.pydata.org/pandas-docs/version/0.18.1/dsintro.html#panel4d-experimental)。