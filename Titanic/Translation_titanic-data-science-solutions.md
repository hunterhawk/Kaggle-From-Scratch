# 泰坦尼克号数据科学解决方案
>[原文链接](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
>该笔记是[Data Science Solutions](
https://www.amazon.com/Data-Science-Solutions-Startup-Workflow/dp/1520545312)一书的配套产品。
>该笔记引导我们完成解决Kaggle等网站数据科学竞赛的典型工作流程。
>已经存在几个优秀的笔记来研究数据科学竞赛的入门。然而，许多笔记会跳过关于如何开发解决方案的一些解释，这是因为这些笔记本是由专家开发，并为专家而作的。本笔记的目标是按照一步步进行的工作流程，解释我们在解决方案开发过程中做出的每个决定的每个步骤和基本原理。

## 工作流程阶段介绍
以[Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)竞赛为例，
解决方案的工作流程经历了[Data Science Solutions](
https://www.amazon.com/Data-Science-Solutions-Startup-Workflow/dp/1520545312)一书中描述的七个阶段。分别如下：

1. 问题或难点的定义
2. 获取训练和测试数据
3. 对数据进行讨论，预处理，清洗
4. 分析，识别模型并探索数据
5. 建模，预测并解决问题
6. 可视化，形成报告，呈现问题的解决步骤和最终解决方案
7. 提供或提交结果

上述的工作流程指示每个阶段的一般顺序。但是也存在一些例外的情况，比如：

* 我们可以结合多个工作流程阶段。我们可以通过可视化数据进行分析。
* 执行早于指示的阶段。我们可能会在讨论前后分析数据。
* 在我们的工作流程中多次执行一个步骤。例如可视化阶段可以多次使用。
* 完全丢弃一个步骤。我们可能不需要供应阶段来进行产品化或服务我们的数据集针对某一比赛。

## 工作流程目标
**分类**。我们可能希望对样本进行识别或分类。我们可能还想了解不同类的含义或不同类与我们的解决方案目标的相关性。

**相关**。我们可以基于训练数据集内的可用特征来解决问题。数据集中的哪些特征对我们的解决方案目标有重大贡献？从统计学上讲，特征和解决方案目标之间是否存在相关性？随着特征值的变化，解决方案状态是否也会发生变化并且反之也成立？这可以针对给定数据集中的数值特征和类别特征进行测试。我们可能还希望确定除后续目标和工作流程阶段的生存之外的特征之间的相关性。关联某些功能可能有助于创建，完善或更正特征。

**转换**。对于建模阶段，我们需要准备数据。根据模型算法的选择，我们可能需要将所有特征转换为数值等效值。例如，将文本分类值转换为数值。

**完整**。数据准备可能还需要我们估计特征中的任何缺失值。当没有缺失值时，模型算法可能最有效。

**纠正**。我们还可以分析给定的训练数据集中的错误或可能在特征内提取值，并尝试纠正这些值或排除包含错误的样本。一种方法是检测我们的样本或特征中的任何异常值。如果某项特征无益于分析，或者可能会严重影响结果的正确性，我们也可能完全丢弃该特征。

**创建**。我们是否可以基于现有特征或一组特征创建新特征，以便新特征遵循关联，转换和完整性目标。

**图表化**。如何根据数据的性质和解决方案目标，来选择正确的可视化图表。

## 重构发布于2017年1月29日
我们基于(a)读者收到的评论，(b)将笔记本从Jupyter内核(2.7)移植到Kaggle内核(3.5)以及(c)审查几个最佳实践内核的问题，对笔记进行了重大的重构。
### 用户评论

* 为某些操作组合训练和测试数据，例如将数据集中的标称值转换为数值。 （感谢@Sharan Naribole）
* 正确观察 - 近30％的乘客有兄弟姐妹和/或配偶。 （感谢@Reinhard）
* 正确解释逻辑回归系数。 （感谢@Reinhard）

### 移植问题

* 指定绘图尺寸，将图例添加到绘图中。

### 最佳实践

* 在项目早期执行特征相关分析。
* 使用多个图而不是叠加图来提高可读性。

## 问题或难点的定义
诸如Kaggle在内的竞赛网站定义要解决的问题，或提出需要解答的问题，同时提供用于训练数据科学模型的数据集，以及根据测试数据集测试模型结果。泰坦尼克号幸存竞赛的问题定义在Kaggle中描述如下：
>根据已知训练数据集列出的在泰坦尼克号灾难中幸存的乘客的一组样本进行学习，我们的模型是否能够基于给定的不包含幸存信息的测试数据集，决策这些位于测试集中的乘客幸存与否。

我们可能还希望对我们问题领域有一些预先的了解，这在[比赛描述页面](https://www.kaggle.com/c/titanic)上有所描述，以下是需要注意的要点：
>1912年4月15日，在其处女航中，泰坦尼克号与冰山相撞后沉没，造成了2224名乘客和机组人员中1502人死亡。幸存率为32％。
>造成此次海难失事的原因之一是乘客和机组人员没有足够的救生艇。
>尽管在巨轮沉没中幸存有一些幸运因素，但有些人比其他人更容易幸存，例如妇女，儿童和上流社会阶层。
```
# 数据分析及讨论
import pandas as pd
import numpy as np
import random as rnd

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# 机器学习
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
```
## 获取数据
Python Pandas包帮助我们处理数据集。我们首先将训练和测试数据集收集到Pandas DataFrames中。我们还组合这些数据集以便在两个数据集上一起运行某些操作。
```
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
```
## 通过描述数据进行分析
Pandas还帮助描述数据集，在我们的项目早期回答以下问题：

**数据集中有哪些特征？**

注意这些特征名称以便直接操作或分析，这些特征名称在[Kaggle数据页面](https://www.kaggle.com/c/titanic/data)中描述。
```
print(train_df.columns.values)
```
```
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
```

**哪些特征是类别的？**

这些值将样本分类为类似样本的集合。在分类特征中，其数值是否基于名义，序数，比率或区间？除此之外，这有助于我们选择适当的可视化图。

* 类别型：Survived, Sex, and Embarked. 序数型: Pclass.


**哪些特征是数值的？**

哪些功能是数值的？这些值随样本而变化。在数值特征中，值是离散的，连续的还是基于时间序列的？除此之外，这有助于我们选择适当的可视化图。

* 连续型: Age, Fare. 离散型: SibSp, Parch.

```
# 数据预览
train_df.head()
```

|  | PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 3 | Braund, Mr.Owen Harris | male | 22.0 | 1 | 0 | A/5 21171 | 7.2500 | NaN | S |
| 1 | 2 | 1 | 1 | Cumings, Mrs.John Bradley(Florence Briggs Th... | female | 38.0 | 1 | 0 | PC 17599 | 71.2833 | C85 | C |
| 2 | 3 | 1 | 3 | Heikkinen, Miss. Laina | female | 26.0 | 0 | 0 | STON/O2. 3101282 | 7.9250 | NaN | S |
| 3 | 4 | 1 | 1 | Futrelle, Mrs. Jacques Heath (Lily May Peel) | female | 35.0 | 1 | 0 | 113803 | 5301000 | C123 | S
| 4 | 5 | 0 | 3 | Allen, Mr. William Henry | male | 35.0 | 0 | 0 | 373450 | 8.0500 | NaN | S |

**哪些特征是混合数据类型？**

同一特征内的数字，字母数值数据。这些是修正目标的候选。

* Ticket是数字和字母数值数据类型的混合。Cabin是字母数值。

**哪些特征可能包含错误或拼写错误？**

对于大型数据集来说，这很难检查，但是从较小的数据集中查看一些样本就可以告诉我们，哪些特征可能需要更正。

* Name特征可能包含错误或拼写错误，因为有多种方法可用于描述姓名，包括标题，圆括号和用于替代或短名称的引号。

```
train_df.tail()
```
![image2](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image2.png)

**哪些功能包含blank，null或empty values？**

这些都需要纠正。

* Cabin > Age > Embarked 训练数据集中这些特征包含多个空值。
* Cabin > Age 在测试数据集中不完整。

**各种特征的数据类型是什么？**

在转换目标时帮助我们。

* 七个特征是整数或浮点数。在测试数据集的情况下为六个。
* 五个特征为字符串（对象）。

```
train_df.info()
print('_'*40)
test_df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
________________________________________
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```

**样本中数值特征值的分布是什么？**

这有助于我们在其他早期见解中确定实际问题域的训练数据集的代表性。

* 总样本是泰坦尼克号上实际乘客人数(2224)的40％(891)。
* Survived是一个具有0或1值的分类特征。
* 大约38％的样本存活，实际存活率为32％。
* 大多数乘客（>75％）没有与父母或孩子一起旅行。
* 近30％的乘客有兄弟姐妹和/或配偶。
* 票价差异很大，很少有乘客（<1％）支付高达512美元。
* 年龄在65-80岁之间的老年乘客（<1％）很少。
```
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
```
![image3](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image3.png)

**分类特征的分布是什么？**

* Name在整个数据集中是唯一的（count = unique = 891）
* Sex变量为两个可能的值，男性为65％（top=男性，freq = 577 / count = 891）
* Cabin值在样本中有几个副本。或者说，几个乘客共用一个客舱小屋。
* Embarked有三个可能的值。大多数乘客使用的S港口（top = S）
* Ticket特征有高比率（22%）的重复值（unique = 681）

```
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`
```
![image4](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image4.png)

```
train_df.describe(include=['O'])
```
![image5](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image5.png)

## 基于数据分析的假设
我们基于迄今为止所做的数据分析得出以下假设。我们可能会在采取适当行动之前进一步验证这些假设。
### 相关
我们想知道每个特征与生存的相关性。我们希望在项目的早期阶段完成这项工作，并将这些快速关联与项目后期的建模关联相匹配。
### 补充完整
1. 我们可能希望补充完整年龄特征，因为它与生存明确相关。
2. 我们可能希望补充完整登船港口特征，因为它还可能与生存或其他重要特征相关联。
### 修正
1. 船票编号特征可能会从我们的分析中删除，因为它包含高比例的重复项（22％），并且票价和生存之间可能没有相关性。
2. 客舱特征可能会被删除，因为其极度不完整或者在训练和测试数据集中都包含许多空值。
3. PassengerId特征可能会从训练数据集中删除，因为它对生存没有贡献。
4. 姓名特征相对不标准，可能无法直接影响生存，因此可能会被删除。
### 创建
1. 我们可能想要创建一个名为Family的基于Parch和SibSp的新特征，以获得船上家庭成员的总数。
2. 我们可能希望设计姓名特征以将Title提取为新特征。
3. 我们可能想为年龄范围创建新特征。这将连续的数字特征转换为序数分类特征（Ordinal categorical feature）。
4. 如果其有助于我们的分析，我们可能还想创建一个票价范围特征。
### 分类
我们还可以根据前面提到的问题描述添加我们的假设。
1. 女性（Sex = female）更有可能幸存下来。
2. 儿童（Age < ？）更有可能幸存下来。
3. 上层乘客（Pclass = 1）更有可能幸存下来。（注：pclass：社会经济地位（SES）的代表）
## 通过旋转特征进行分析
为了确认我们的一些观察和假设，我们可以通过相互转动特征来快速分析我们的特征相关性。在此阶段我们只能为没有任何空值的特征执行此操作。对于分类（Sex），序数（Pclass）或离散（SibSp，Parch）类型的特征，这样做也是有意义的。

* **Pclass** 我们观察到Pclass = 1和Survived之间的显著相关性（> 0.5）**（分类#3）**。我们决定在我们的模型中包含此特征。
* **Sex** 我们在问题定义中确认Sex=female的生存率非常高，为74％**（分类#1）**。
* **SibSp和Parch** 这些特征对于某些值具有零相关性。最好从这些单独的特征中导出一个特征或一组特征**（创建#1）**。
```
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image6](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image6.png)
```
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image7](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image7.png)
```
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image8](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image8.png)
```
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image9](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image9.png)
## 通过可视化数据进行分析
现在我们可以继续使用可视化分析数据来确认我们的一些假设。
### 关联数值特征
让我们首先了解数值特征与我们的解决方案目标（存活）之间的相关性。

直方图可用于分析像Age这样的连续数值变量，其中条带或范围将有助于识别有用的模式。直方图可以使用自动定义的区间或等距离范围来指示样本的分布。这有助于我们回答与特定范围相关的问题（例如：婴儿的生存率是否更高？）

请注意，直方图可视化中的x轴表示样本或乘客的数量。

**观察**

* 婴儿（Age <= 4）的存活率很高。
* 最年长的乘客们（Age = 80岁）幸免于难。
* 大量15-25岁的人没有活下来。
* 大多数乘客年龄在15-35岁之间。

**决策**

这个简单的分析证实了我们作为后续工作流程阶段决策的假设。

* 我们应该在模型训练中考虑年龄特征。**（我们的假设分类#2）**
* 补充完整年龄特征的那些空值。**（补充完整#1）**
* 我们应该将年龄特征进行范围分组。**（创建#3）**
```
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
```
![image10](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image10.png)
### 关联数值和有序特征
我们可以使用单个图组合多个特征来识别相关性。这可以通过具有数值的数值特征和分类特征来完成。

**观察**

* Pclass = 3 有大多数乘客，但大多数人没有幸存。确认了我们的**分类假设#2**。
* Pclass = 2和Pclass = 3的婴儿乘客大部分幸存下来。进一步限定了我们的**分类假设#2**。
* Pclass = 1的大多数乘客幸免于难。确认了我们的**分类假设#3**。
* Pclass在乘客年龄分布方面有所不同。

**决策**

* 考虑使用Pclass进行模型训练。
```
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
```
![image11](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image11.png)
### 关联类别特征
现在我们可以将分类特征与我们的解决方案目标相关联。

**观察**

* 女性乘客的生存率远高于男性。**(分类假设#1)**
* 在Embarked = C中的例外情况，其中雄性具有更高的存活率。这可能是Pclass和Embarked之间的相关性，反过来是Pclass和Survived，不一定是Embarked和Survived之间的直接相关。
* 与C、Q港口的Pclass = 2相比，男性在Pclass = 3时的存活率更高。**(补充完整假设#2)**
* 登船口岸的Pclass = 3和男性乘客的生存率各不相同。**(相关假设#1)**

**决策**

* 将性别特征添加到模型训练中。
* 补充完整并添加登船口岸特征以进行模型训练。

```
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
```
![image12](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image12.png)
### 关联类别和数值特征
我们还可以将分类特征（使用非数值的值）和数值特征相关联。 我们可以考虑将登船口岸（分类非数值特征），性别（分类非数值特征），票价（数值连续特征）与幸存（分类数值特征）相关联。

**观察**

* 船票付费较高的乘客有更好的幸存率。这确认我们 **创建假设(#4)** 关于票价范围的假设。
* 登船口岸与幸存率相关。**相关假设(#1)和补充完整假设(#2)**

**决策**

* 考虑票价范围特征。

```
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
```
![image13](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image13.png)
## 讨论数据
我们收集了有关数据集和解决方案要求的若干假设和决策。 到目前为止，我们没有必要更改单个功能或值来实现这些功能。 现在让我们执行我们的决策和假设，以纠正，创建和完成目标。
### 通过丢弃特征进行修正
这是一个很好的起始目标。通过删除特征，我们处理的数据点更少。加速我们的笔记本电脑并简化分析。

根据我们的假设和决策，我们希望丢弃Cabin **(修正#2)** 和Ticket **(修正#1)** 特征。

请注意，在适用的情况下，我们同时对训练数据集和测试数据集执行操作以保持一致。
```
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
```
>Before (891, 12) (418, 11) (891, 12) (418, 11)
>('After', (891, 10), (418, 9), (891, 10), (418, 9))

### 从现有特征提取创建新特征
我们想要分析是否可以设计Name特征来提取Title(头衔)并测试Title(头衔)和生存之间的相关性，然后再删除Name和PassengerId功能。

在下面的代码中，我们使用正则表达式提取标题功能。 RegEx模式 **(\w+\.)** 匹配Name特征中以.字符结尾的第一个单词。expand = False标志返回一个DataFrame。

**观察**

当我们绘制Title，Age和Survived时，我们注意到以下观察结果。

* 大多数头衔准确地将年龄组带入例如：主标题的年龄均值为5年。
* Title年龄段的生存率略有不同。
* 某些头衔大多存活下来(Mme，Lady，Sir)，而某些大多没有存活下来(Don，Rev，Jonkheer)。

**决策**

* 我们决定保留新的Title(头衔)特征以进行模型训练。

```
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```
![image14](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image14.png)
我们可以用更常见的Name替换许多Title(头衔)或将它们归类为Rare(稀少)。
```
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
```
![image15](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image15.png)
我们可以将类别标题转换为序数。
```
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
```
![image16](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image16.png)
现在我们可以安全地从训练数据集和测试数据集中删除Name特征。我们也不需要训练数据集中的PassengerId特征。
```
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
```
>((891, 9), (418, 9))

### 转换类别特征

现在我们可以将包含字符串的特征转换为数值。这是大多数模型算法所必需的。这样做也将有助于我们实现特征补充完整的目标。

让我们首先将性别特征转换为名为Gender的新特征，其中 female = 1，male = 0。
```
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
```
![image17](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image17.png)

### 补充完整数值连续特征

现在我们应该开始估计和补充缺少值(missing)或空值(null)的特征。我们将首先为Age特征执行此操作。

我们可以考虑三种方法来补充数值连续特征。

1. 一种简单的方法是在均值和标准差之间生成随机数。
2. 猜测缺失值的更准确方法是使用其他相关特征。 在我们的例子中，我们注意到Age，Gender和Pclass之间的相关性。使用Pclass和Gender特征组合的集合的中值来猜测Age年龄值。因此，统计(Pclass=1, Gender=0)，(Pclass=1, Gender=1)的Age年龄中值以及更多……
3. 结合上述方法1和2。因此，不是基于中位数来猜测年龄值，而是根据Pclass和Gender组合的集合使用均值和标准差之间的随机数。

方法1和3会将随机噪声引入我们的模型。多次执行的结果可能会有所不同。我们更喜欢方法2。
```
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
```
![image18](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image18.png)
我们首先准备一个空数组，以包含基于Pclass 与 Gender组合而得到的Age猜测值。
```
guess_ages = np.zeros((2,3))
guess_ages
```
>array([[0., 0., 0.],
>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0., 0., 0.]])

现在我们迭代Sex(0或1)和Pclass(1,2,3)来计算六种组合的Age的猜测值。
```
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
```
![image19](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image19.png)

创建年龄段并确定与幸存的相关性。
```
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```
![image20](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image20.png)

根据这些范围将年龄值替换为序数。
```
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
```
![image21](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image21.png)

我们现在可以删除AgeBand特征。
```
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
```
![image22](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image22.png)

### 组合现有特征，创建新特征
我们可以为FamilySize创建一个新特征，它结合了Parch和SibSp。这将使我们能够从数据集中删除Parch和SibSp特征。
```
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image23](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image23.png)

我们可以创建另一个名为IsAlone的特征。
```
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
```
![image24](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image24.png)

让我们删除Parch，SibSp和FamilySize特征，转而使用IsAlone特征。
```
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
```
![image25](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image25.png)

我们还可以创建一个结合Pclass和Age的人工特征。
```
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
```
![image26](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image26.png)

### 补充完整类别特征

登船口岸特征根据登船港口获取S，Q，C值。我们的训练数据集有两个缺失值。我们简单地用最常见的值补充这两个值。
```
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
```
>'S'

```
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```
![image27](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image27.png)

### 将类别特征转换为数值特征

我们现在可以通过创建新的数值登船口岸特征来转换EmbarkedFill特征。
```
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
```
![image28](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image28.png)

### 快速补充并转换数值特征

现在，我们可以使用模型为测试数据集中的单个缺失值完成“票价”功能，以获取此特征最常出现的值。我们在一行代码中完成此操作。

请注意，由于我们只替换单个值，因此我们不会创建中间新特征或进行任何进一步的相关分析以猜测缺失特征。 该补充目标实现了模型算法对非空值进行操作的期望要求。

我们可能还希望将票价四舍五入到两位小数，因为它代表货币。
```
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
```
![image29](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image29.png)
现在我们可以创建票价范围特征。
```
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
```
![image30](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image30.png)

根据FareBand将票价特征转换为序数值。
```
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

train_df.head(10)
```
![image31](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image31.png)

对测试数据集进行同样的操作。
```
test_df.head(10)
```
![image32](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image32.png)
## 建模，预测和解决(问题)
现在我们准备训练模型并预测所需的解决方案。 有60多种预测建模算法可供选择。 我们必须了解问题的类型和解决方案要求，以缩小到我们可以评估的少数几个模型。 我们的问题是分类和回归问题。 我们想要确定输出（幸存与否）与其他变量或特征（性别，年龄，港口......）之间的关系。 我们还使用了一种机器学习方法，称为监督学习，因为我们正在使用给定的数据集训练我们的模型。 有了这两个标准 - 监督学习加分类和回归，我们可以将我们选择的模型缩小到几个。 这些包括：

* Logistic回归
* KNN 或  K近邻
* 支持向量机
* 朴素贝叶斯分类器
* 决策树
* 随机森林
* 感知机
* 人工神经网络
* RVM或相关向量机

```
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
```
>((891, 8), (891,), (418, 8))

### Logistic回归
Logistic回归是在工作流程中尽早使用的有用模型。 Logistic回归通过使用逻辑函数（累积逻辑分布，cumulative logistic distribution）估计概率来测量分类因变量（特征）与一个或多个自变量（特征）之间的关系。[Logistic regression wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)

请注意模型基于我们的训练数据集生成的置信度分数。

```
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
```
>80.36

我们可以使用Logistic回归来验证我们对特征创建和完成目标的假设和决策。 这可以通过计算决策函数中特征间的系数来完成。

正系数增加了响应的对数几率(log-odds)（从而增加了概率），负系数降低了响应的对数几率（从而降低了概率）。

* 性别是最高的正系数，暗示随着性别值的增加（male：0到female：1），Survived= 1的概率增加最多。
* 相反，随着Pclass的增加，Survived=1的概率降低最多。
* 由此可以看出，Age * Class 是一个很好的人工模型，因为它与Survived具有第二高的负相关。
* Title(头衔)是第二高正相关。

```
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```
![image33](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image33.png)

### SVM支持向量机
接下来，我们使用支持向量机进行建模，支持向量机是监督学习模型，具有相关的学习算法，用于分析分类和回归分析。给定一组训练样本，每个训练样本被标记为属于两个类别中的一个或另一个，SVM训练算法构建将新测试样本分配给一个类别或另一个类别的模型，使其成为非概率二元线性分类器。 [SVM_wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine)

请注意，该模型生成的置信度得分高于Logistic回归模型。
```
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```
>83.84

### KNN
在模式识别中，k-Nearest Neighbors算法（或简称k-NN）是用于分类和回归的非参数方法。 样本按其邻居的多数票进行分类，样本被分配给其k个最近邻居中最常见的类（k是正整数，通常很小）。 如果k = 1，则简单地将对象分配给该单个最近邻居的类。 [K-nearest_neighbors_algorithm_wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

KNN置信度得分优于Logistic回归，但比SVM差。
```
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```
>84.74

### 朴素贝叶斯分类
在机器学习中，朴素贝叶斯分类器是一系列简单的概率分类器，它基于贝叶斯定理应用特征之间的强（朴素）独立假设。 朴素贝叶斯分类器具有高度可扩展性，在学习问题中需要多个变量（特征）的线性参数。 [Naive_Bayes_classifier_wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

该模型生成的置信度得分是迄今为止评估的模型中最低的。
```
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```
>72.28

### 感知机
感知器是用于二元分类器的监督学习算法（其功能可以判断由数字向量表示的输入是否属于某个特定类）。 它是一种线性分类器，即一种分类算法，其基于将一组权重与特征向量组合的线性预测函数进行其预测。 该算法允许在线学习，因为它一次一个地处理训练集中的元素。[Perceptron_wikipedia](https://en.wikipedia.org/wiki/Perceptron)
```
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
```
>78.0

### 线性支持向量分类
```
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
```
 >79.12

### 随机梯度下降
```
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```
>78.56

### 决策树
该模型使用决策树作为预测模型，将特征（树枝）映射到关于目标值（树叶）的结论。 目标变量可以采用有限值集的树模型称为分类树; 在这些树结构中，叶子表示类标签，分支表示导致这些类标签的特征的连接。 目标变量可以采用连续值（通常是实数）的决策树称为回归树。[Decision_tree_learning_wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

到目前为止评估的模型中该模型置信度得分最高。
```
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```
>86.76

### 随机森林
下一个模型随机森林是最受欢迎的模型之一。 随机森林或随机决策森林是用于分类，回归和其他任务的集成学习方法，其通过在训练时构建多个决策树（n_estimators = 100）并输出作为类的模式的类（分类）来操作。 或者各树的预测（回归）平均值。 [Random_forest_wikipedia](https://en.wikipedia.org/wiki/Random_forest)

到目前为止评估的模型中模型置信度得分最高。 我们决定使用此模型的输出（Y_pred）来创建竞赛结果提交。
```
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```
>86.76

## 模型评估
我们现在可以对所有模型进行评估，以便为我们的问题选择最佳模型。 虽然决策树和随机森林得分相同，但我们选择使用随机森林来纠正决策树过拟合的缺陷。
```
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
```
![image34](https://github.com/hunterhawk/Kaggle-From-Scratch/blob/master/Titanic/res/image34.png)
```
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)
```
我们向竞赛网站Kaggle提交的结果为6,082个参赛作品中的3,883个得分。 在比赛开始时，这个结果是指示性的。 此结果仅占提交数据集的一部分。 对我们的第一次尝试来说不错。

## 参考

该笔记是基于解决Titanic Competition的工作以及其他资源而创建的。

- [A journey through Titanic](https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
-  [Getting Started with Pandas: Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/details/getting-started-with-random-forests)
-  [Titanic Best Working Classifier](https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier)
