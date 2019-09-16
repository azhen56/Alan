# 画图中文和负号显示异常问题
"""
(将C:\Windows\Fonts文件夹下的simhei.ttf(黑体)文件
 复制到C:\ProgramData\Anaconda3\Lib\site-packages\matplotlib\mpl-data\fonts\ttf文件夹下)
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='simhei', size=12)
matplotlib.rcParams['axes.unicode_minus'] = False
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

os.chdir("D:/maze")  # 更改路径
os.getcwd()  # 查询当前路径
df = pd.read_csv(file_path, sep=',', parse_dates=['live_dt', 'observe_dt_inv', 'etl_update_time'],
                 dtype={"htl_cd":int,'otb':float, 'inv':float,'adv_days':float})
#######################################################
# 数据库中文插入错误用下面两句
import os
os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"

#######################################################
# 唯一值个数
data.htl_cd.nuique()
# 填充缺失值
df.fillna(method="ffill")  # 将空值填充为上一个非空值
df.fillna(method="backfill")  # 将空值填充为下一个非空值
df.interpolate()  # 将空值填充为上下两个值的均值
# 日期处理问题(标准化日期)
print(time.strftime('%Y-%m-%d',time.localtime())) #输出当前时间
data["d_insert"] = datetime.datetime.now()  # 当前时间

# 填充缺失值 '1900-01-01'， '0001-01-01' -> '1900-01-01'
df["max_grant_date"] = df["max_grant_date"].map(
    lambda x: datetime.datetime(1900, 1, 1, 0, 0)
    if x == datetime.datetime(1, 1, 1, 0, 0) or pd.isnull(x)
    else x)
# 日期加减
(datetime.datetime(2019,5,3)-datetime.datetime(2017,1,1)).days
data['adv_day'] = (data.live_dt - data.order_dt).dt.days
today-datetime.timedelta(days=i)
df['observe_date'] = df['live_dt'] - pd.to_timedelta(df['adv_days'], unit='day')

data['月_日'] = data.日期.dt.strftime('%m-%d')

#######################################################
# 列出有空值的列
df.columns[df.isnull().any()].tolist()

# 统计有空值的列的空值数
df[df.columns[df.isnull().any()]].isnull().sum()

# 将字符代码替换为中文
df["dict_sex"] = df["dict_sex"].map(lambda x: "女" if x == "FG00101" else "男")
df["custgroup"] = df["custgroup"].map({"MD010100": "积极精英", "MD010101": "聪明消费"}).fillna(0)

# 将中文转为数字(数字不区分大小)
df["dict_prd3"] = df["dict_prd3"].astype("category").cat.codes

# 按某两列排序
# 按照class升序，score降序
df.sort_values(['class','score'],ascending=[1,0],inplace=True)
grouped = df.groupby(['class']).head(2) #取每个class的前2行数据

month = sorted(data.data_date.unique())  # 升序
month = sorted(data.data_date.unique(), reverse=True)  # 降序
# 用 operator 函数来加快速度
from operator import itemgetter, attrgetter
sorted(df, key=itemgetter(1, 2))

#计算每个订单占总订单的比例
df["Percent_of_Order"] = df["ext price"] / df.groupby('order')["ext price"].transform('sum')
#增加一列分组后的总和
data['aa'] = data.groupby(['htl_cd','adv_days'])['delta'].transform('sum')
#增加两列
data[['aa','bb']] = data.groupby(['htl_cd','adv_days'])['delta','week_std'].transform('sum')

# 列表拼接成字符串
str_1 = '~'.join(list_1)
#######################################################
cols = [
    "age",
    "aum_bal_avg",
    "trad_amt_in",
    "trad_cnt_in",
    "trad_amt_out",
    "trad_cnt_out"]
t = 0.01
i = 5

# 设置阈值(上界99分位,下界1分位)
def threshold(df, val, t, k):
    df[val] = np.clip(df[val],np.percentile(df[val],t),np.percentile(df[val],k))
    return df
for val in cols:
    df = threshold(df, val, 1, 99)
tmp = data.groupby(['htl_cd'])['delta'].quantile([0.5,0.75,0.9]).unstack() #20/75/90分位

#转为e**(1+x),且上下界为0,100
df["unit_sales"] = np.clip(np.expm1(df["unit_sales"]), 0, 100)

#分类阈值
from sklearn.preprocessing import binarize
y_pred_class = binarize([y_pred_prob], 0.3)[0] #大于0.3的为1,否则则为0

#######################################################
# 新增客户
df = pd.merge(data_7, data_8, how="outer", on="客户ID", indicator=True).query(
    '_merge=="right_only"'
)
df = df[["客户ID", "客户类别_y"]]
print("8月新增客户的数量:" + str(df.shape[0]))

#######################################################
# 多个图组合
g = sns.PairGrid(
    df,
    x_vars=["custgroup", "age"],
    y_vars=["aum_bal_avg", "trad_cnt_in"],
    aspect=1.6,
    size=4.6)
g.map(sns.barplot, palette="pastel")
g.savefig(r"D:\maze\fw\feature_analysis/0005.png")

# 透视图
ptResult = df.pivot_table(
    values=["cust_id"], index=["age"], columns=["gender"], aggfunc=[np.size]
)
print(ptResult)

# 包含3个特征的箱图
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()
sns.boxplot(x="gender", y="zonghe_limit", hue="edu_level", data=df)
fig.savefig("D:/maze/fw/00.png", dpi=100)

# 包含3个特征的小提琴图(hue选项只有两个类别值才能加入split=True)
fig = plt.figure()
sns.violinplot(x="edu_level", y="zonghe_limit", hue="gender", data=data2, split=True)
fig.savefig("D:/maze/fw/00.png", dpi=100)

# 画图
# kind : 可选：point 默认, bar 柱形图, count 频次, box 箱体, violin 提琴, strip 散点，swarm 分散点
a = sns.factorplot(
    x="time",
    y="total_bill",
    hue="smoker",
    col="day",
    data=tips,
    kind="box",
    size=9,
    aspect=1.2,
)  # (size越大图越大,aspect越大图越宽)
a.savefig("D:/maze/00.png")

# 预测聚类AUM值
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_aum = pd.read_csv("aum_max_std_3m.csv")
y_aum = pd.read_csv("aum_y.csv")
data_aum = X_aum.merge(y_aum, on="cust_id", how="left")
data_aum.dropna(how="any", inplace=True)  # 删除有空值的行
X_aum = data_aum[["aum_bal_max", "aum_bal_std"]]
y_aum = data_aum["aum_bal_max_aft"]
X_train, X_test, y_train, y_test = train_test_split(X_aum, y_aum, train_size=0.8)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().transform(X_test)
clf = LinearRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

#######################################################
# 年龄分层
bins = [0 , 3 , 7 , 14 , 100]  #左开右闭
labels = ["<=3", "4_7", "8_14",'15以上']
df["age_cut"] = pd.cut(df.age, bins, labels=labels)

# 除去'MD010100','MD010101','MD010198','MD010104'类客户
df = df[~df["客户类别"].isin(["MD010100", "MD010101", "MD010198", "MD010104"])]

# 对日期排序，从小到大
limit_bal_df = (
    df.groupby(["cust_id"])
    .apply(lambda x: x.sort_values(["data_date"], ascending=True))
    .reset_index(drop=True)
)

# 数据按行 0-1化
from sklearn.preprocessing import Normalizer
df = Normalizer(norm="l1").fit_transform(data)  # 样本各个特征值除以各个特征值的绝对值之和

# 读取txt文件(文件名中文)
data = pd.read_table(open("E:/data/AA/红酒品质数据.txt"))

# 通过stack将一行一个id多列多个产品，转为多行多行多个重复ID不同产品
df = pd.read_table(open("E:/data/AA/购物篮数据.txt"), sep=",")
data = df.iloc[:, 7:]
data.index=df.cardid
data = data.stack().reset_index()
data.rename(columns={"level_1": "prd", 0: "score"}, inplace=True)
df = data[data.score == 1]

# 将空值替换为类别数量最多的一类*(需要将字符型类别转为数字型)
from scipy.stats import mode
mode_embarked = mode(df.Embarked)[0][0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)

# 将一列转多列0_1型数据
pd.get_dummies(df["Embarked"], prefix="Embarked")
df_final = pd.get_dummies(df,['purpose'],drop_first=True)

# 用Pclass类别的Fare均值 替换 同一Pclass类别但Fare缺失的值
fare_means = df.groupby(["Pclass"])["Fare"].mean()
df_test["Fare"] = df_test[["Fare", "Pclass"]].apply(
    lambda x: fare_means[x["Pclass"]] if pd.isnull(x["Fare"]) else x["Fare"], axis=1)

# 将第二列调到第一列
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

#######################################################
# 随机森林网格搜索最优参数
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

parameter_grid = {"max_features": [0.5, 1.], "max_depth": [5., None]}
grid_search = GridSearchCV(
    RandomForestClassifier(n_estimators=100), parameter_grid, cv=5, verbose=3
)
grid_search.fit(train_data[0:, 2:], train_data[0:, 0])
grid_search.grid_scores_
sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
grid_search.best_score_
grid_search.best_params_
model = RandomForestClassifier(n_estimators=100, max_features=0.5, max_depth=5.0)
model = model.fit(train_data[0:, 2:], train_data[0:, 0])

# 支持向量机网格搜索最优参数
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

parameter_grid = {"C": [1., 10.], "gamma": [0.1, 1.]}
grid_search = GridSearchCV(SVC(kernel="linear"), parameter_grid, cv=5, verbose=3)
grid_search.fit(train_data[0:, 2:], train_data[0:, 0])
sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
grid_search.best_score_
grid_search.best_params_
model = SVC(kernel="linear", C=1.0, gamma=0.1)
model = model.fit(train_data[0:, 2:], train_data[0:, 0])

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
svc = svm.SVC()
grid_search = GridSearchCV(svc, parameters)
grid_search.fit(iris.data, iris.target)
sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
print(grid_search.best_score_)
grid_search.best_params_
model = SVC(kernel="linear", C=1.0, gamma=0.1)

#######################################################
# 产生一列时间变量
data['occ'] = [random.randint(50,200) for _ in range(20)] #随机产生20个整数
rng = pd.date_range("2017-06-01", periods=5, freq="D")
rng = pd.date_range("2017-06-01", periods=5, freq="M")  # 月末最后一天
a = pd.DataFrame(rng)

# 选择年龄在30-32之间的数据
df = df[(df.Age < 32) & (df.Age > 30)]
# df=df[(df.Age<32)|(df.Age>30)]

# 新增一列,并按照其他列的值填充特定值
data["dict_group_type"] = np.nan
data["dict_group_type"] = data["custgroup"].map(
    lambda x: "1"
    if x in ["MD010100", "MD010101", "MD010102"]
    else "2"
    if x in ["MD010103", "MD010104"]
    else "0"
    if x in ["MD010105", "MD010106"]
    else x
)

# 笛卡尔积
rule = ("MD010104 MD010105 MD010106 MD010107 MD010108").split(" ")
df = [(a, b) for a in rule for b in rule]
df = pd.DataFrame(df, columns=["custgroup_1", "custgroup_3"])

# 两两配对(不重复)
import itertools

list(itertools.combinations_with_replacement([1, 2, 3, 4, 5], 2))


# 计算迁移比例('上月客群','当月客群','上月-当月月迁移人数'3列)
def ratio(data):
    num_sum = data.groupby([data.columns.tolist()[0]])[data.columns.tolist()[2]].sum()
    data["ratio"] = data[data.columns.tolist()[2]] / data.apply(
        lambda x: num_sum[x[data.columns.tolist()[0]]], axis=1
    )
    return data


# 一列含有特定字符的选项
is_prd3 = df["prd"].str.contains("prd3")
df = df[is_prd3]

#######################################################
#折线图横坐标为字符型时显示问题
a=tmp2[['int_rate_cut','num','pred']].groupby('int_rate_cut').mean()
a.plot(figsize=(10,4))
plt.xticks(range(len(a.index)),(a.index))
plt.xlabel('时间段')
plt.ylabel('人数')
plt.title('39路车不同时间段人流量预测效果对比')

# 画折线图
f, ((ax1, ax2)) = plt.subplots(1, 2)
cmap = plt.cm.coolwarm

by_credit_score = df.groupby(["year", "grade"]).loan_amount.mean()
by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)
ax1.set_title("Loans issued by Credit Score", fontsize=14)

by_inc = df.groupby(["year", "grade"]).interest_rate.mean()
by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)
ax2.set_title("Interest Rates by Credit Score", fontsize=14)

ax2.legend(
    bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1),
    loc=5,
    prop={"size": 12},
    ncol=7,
    mode="expand",
    borderaxespad=0.,
)

#######################################################
# 将分类型变量转换为数值形变量
from sklearn.preprocessing import LabelEncoder

for col in data.columns:
    data[col] = LabelEncoder().fit_transform(data[col])

#######################################################
# 特征组合
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# from (X_1, X_2) to (1, X_1, X_2, X_1^2, X_1X_2, X_2^2)
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(2)
poly.fit_transform(X)

# from (X_1, X_2, X_3) to (1, X_1, X_2, X_3, X_1X_2, X_1X_3, X_2X_3, X_1X_2X_3)
X = np.arange(9).reshape(3, 3)
poly = PolynomialFeatures(degree=3, interaction_only=True)  # 只做特征之间乘积
poly.fit_transform(X)

#######################################################
# 生成等差数列
import itertools
list1=[]
gen = itertools.takewhile(lambda n: n < 6, itertools.count(1, .5))
for i in gen:
    list1.append(i)
    
#######################################################
# 计算各列数据增长百分比
a = df10.groupby(["year", "purpose"])["purpose"].count().unstack('year').fillna(0)
a["zong"] = a.sum(axis=1)
a.diff().iloc[1:].reset_index(drop=True) / a.iloc[:-1].reset_index(drop=True)

#处理差分后的不合适的行
data_h55 = data_h55.sort_values(by=['htl_cd','live_dt','observe_dt_inv'])
data_h55.loc[data_h55.groupby(['htl_cd','live_dt']).head(1).index,'add_otb']=0

# 统计一个数组各值得个数
pd.value_counts(clf.labels_, sort=False)

#######################################################
# 异常不中断
s1 = "hello"
try:
    int(s1)
except Exception as e:
    print("e????")

try:
    int(s1)
except Exception as e:
    print("ee")



# 异常中断
try:
    int(s1)
except Exception as e:
    pass
    tmp = 1
    if tmp == 1:
        break
    print('asda')
    # print("e")
    # print(e)
    # raise

try:
    int(s1)
except Exception as e:
    print("ee")


#######################################################
# 0和1互换
def f(x):
    return 1 if x == 0 else 0

pred = [i for i in map(f, pred)]

#######################################################
# 选择特定列
no_columns = ["AA", "BB", "CC"]
columns = ["AA", "BB", "CC", "DD", "EE"]
use_columns = X_train[[i for i in columns if i not in no_columns]]

#######################################################
# 特征选择
# 互信息
from minepy import MINE
m = MINE()
x = np.random.uniform(-1, 1, 10000) #随机产生10000个[-1,1]之间的小数
m.compute_score(x, x**2)
print( m.mic())

# 随机森林特征重要性排名
a = pd.DataFrame(model.best_estimator_.feature_importances_,columns=['importance'],index = data.columns[1:])
a.sort_values(by='importance',ascending=True).iloc[-8:].plot.barh(figsize=(6,3))
print(a.sort_values(by='importance',ascending=False))

# 计算各列与y值得pearson相关系数
from scipy.stats import pearsonr

columns = X_train.columns
feature_importance = [
    (column, pearsonr(X_train[column], y_train)[0]) for column in columns
]
feature_importance.sort(key=lambda x: x[1])

# xgb 特征选择
feature_importance = model.get_booster().get_fscore()  #GPU版本
feature_importance = model.get_score()                 #CPU版本
feature_importance = sorted(
    feature_importance.items(), key=lambda x: x[1], reverse=True
)

# l1,l2 特征选择
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty="l1", random_state=0, n_jobs=-1).fit(X_train, y_train)
pred = lr.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, pred))
feature_importance = [(i[0], i[1]) for i in zip(columns, lr.coef_[0])]
feature_importance.sort(key=lambda x: np.abs(x[1]), reverse=True)

# RandomizedLasso 特征选择
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston

boston = load_boston()
X = boston["data"]
y = boston["target"]
names = boston["feature_names"]
lr = RandomizedLasso(alpha=0.025).fit(X, y)
feature_importance = sorted(zip(names, lr.scores_), key=lambda x: x[1], reverse=True)

# RFE 特征选择(逐个剔除)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rfe = RFE(rf, n_features_to_select=1, verbose=1)
rfe.fit(X_train, y_train)
feature_importance = sorted(
    zip(map(lambda x: round(x, 4), rfe.ranking_), columns), reverse=True
)

# 稳定性选择(统计某个特征被认为是重要特征的频率)
from sklearn.linear_model import RandomizedLasso
from sklearn.datasets import load_boston
boston = load_boston()
# Data gets scaled automatically by sklearn's implementation
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]
rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)
print(sorted(zip(map(lambda x: round(x, 4), rlasso.scores_),names), reverse=True))

#######################################################
# 多数据集连接
from functools import reduce
list1 = [data, age, aum_avg, loan_bal, tradin, tradout, open_date, used, ovd_cnt, tradaum]
df = reduce(lambda x, y: x.merge(y, how='left', on='cust_id'), list1)

#######################################################
#单位销量进行log(1+x)处理
train1['unit_sales'] =  train1['unit_sales'].apply(np.log1p) #logarithm conversion
#销量还原
predict['unit_sales'] = predict['unit_sales'].apply(np.expm1) 
#新增一列星期几--The day of the week with Monday=0, Sunday=6
train1['date']=pd.to_datetime(train1['date'])
train1['dow'] = train1['date'].dt.dayofweek+1
test['year'],test['week'],test['day']=list(zip(*test.date.apply(lambda x: x.isocalendar())))
#提取年月日
data['Year'] = pd.DatetimeIndex(data['date']).year
data['Month'] = pd.DatetimeIndex(data['date']).month
data['Day'] = pd.DatetimeIndex(data['date']).day

#######################################################
#ARIMA自动确定p,q值
import statsmodels.tsa.stattools as st 
order = st.arma_order_select_ic(ts,max_ar=10,max_ma=10,ic=['aic', 'bic', 'hqic']) 
order.bic_min_order

#######################################################
#堆积柱状图
plt.style.use('seaborn-white')
# plt.style.use('dark_background')
type_cluster = stores.groupby(['type','cluster']).size()
type_cluster.unstack().plot(kind='bar',stacked=True,colormap= 'PuBu',figsize=(13,11),grid=False)
plt.title('Stack', fontsize=18)
plt.ylabel('Coun', fontsize=16)
plt.xlabel('Stor', fontsize=16)
plt.show()

#######################################################
#找出2-100中的质数
for i in range(2,101):
    j = 2
    while(i%j != 0):
        j +=1
    if j == i:
        print(i)

#找出4-100中的合数
list=[]
for i in range (4,100):
  for j in range(2,i):
    if(i%j==0):
      list.append(i)
      break
  #else:
    #break
    #list.append(i)
print('以下打印合数：')
print(list)

#######################################################
#对字典排序(按值升序,若值相等,按照键倒序)
dic = {'a':0, 'bc':5, 'c':0, 'asd':4, 'aa':74, 'd':0,'e':0}
dic2=sorted(dic.items(),key=lambda d:d[0],reverse=True)
sorted(dict(dic2).items(),key=lambda d:d[1])

#######################################################
#一个列表，里面是（key, value) 这样的键值对元组，将它转成一个字典对象，并将key相同的value作为一组
data = [("p", 1), ("p", 2), ("p", 3),
        ("h", 1), ("h", 2), ("h", 3)]
result={}
for (key,value) in data:
    result.setdefault(key,[]).append(value)
    
from collections import defaultdict
result = defaultdict(list)
for (key, value) in data:
    result[key].append(value)
#######################################################
#一个环形公路n个加油站,一辆油箱无限容量卡车,
def runningCircle(n,limit,cost):
    for start in range(n):
        #剩余油量
        remainOil=0
        for num in range(n):
            #加油站标号
            index=(num+start)%n 
            #加油
            remainOil=limit[index]+remainOil
            #耗油是否能坚持到下个加油站
            remainOil=remainOil-cost[index]
#            print(remainOil)
            if remainOil<0:
                break
        #检测到通过则返回退出
        if remainOil>=0:
            return start+1
    return '无解'
limit=[1,2,3,4,5,6,7,8,9,10]
cost= [1,1,1,1,1,46,1,1,1,1]
n=len(limit)
runningCircle(n,limit,cost)
#######################################################
#逻辑回归相关实现代码
#sigmoid函数
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度下降函数
def gradAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001    #学习率
    maxCycles=500  #最大迭代次数
    weights=np.ones((n,1)) #初始化权重theata
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights
#######################################################
#小数转百分数
'%.2f%%'%(x*100)
#数据加千分位符号
'{:,}'.format(10000)

#######################################################
#计算余弦相似度
def sim_cos(x, y):
    num = sum([x[i]*y[i] for i in range(len(x))])
    den = sum([x[i]**2 for i in range(len(x))])**0.5*sum([y[i]**2 for i in range(len(y))])**0.5
    return num / den

# 修正余弦相似度
def sim_acs(x, y):
    m_x = sum(x) / len(x)
    m_y = sum(y) / len(y)
    x = [i-m_x for i in x]
    y = [j-m_y for j in y]
    return sim_cos(x, y)

#######################################################    
#返回i天前日期
import datetime
def getYesterday(i): 
    today=datetime.date.today() 
    yesterday=today-datetime.timedelta(days=i)
    return yesterday
yesterday = getYesterday(365)

#######################################################
#读取文件夹下所有.csv文件
import glob
filecsv=glob.glob(r'E:\git\RMS_newfeatures\script\mat_bushu/*.txt')
holiday=[]
for i in filecsv:
    tmp=pd.read_csv(i)
    holiday.append(tmp)
holiday=pd.concat(holiday).reset_index(drop=True)
    
#######################################################
end_day = (datetime.date.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d')
#获取两个日期间的所有日期
def getEveryDay(begin_date,end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list
#返回几号/周几/几月
for target in ['day', 'dayofweek','month']:
    df[target] =df.live_dt.apply(lambda x: getattr(x, target))

####################################################### 
#使用0代替数组x中的nan元素，用有限的数字代替inf元素
np.nan_to_num(train_y.values)

#######################################################
#shift()操作对数据进行移动，可以观察前一天和后天是不是节假日
date['prev_day_is_holiday'] = date['is_holiday'].shift().fillna(0)
date['next_day_is_holiday'] = date['is_holiday'].shift(-1).fillna(0)

#######################################################
#画一个分类变量的条形图/饼图
xw = pd.get_dummies(data['workclass']).sum(axis = 0) #转换为哑元再求和
fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(12,5)) #两个图的排列
xw.plot(kind='bar',ax=axes[0])
xw.plot(kind='pie',ax=axes[1])

#######################################################  
#保存模型
import pickle
with open(f'rf.pickle.dat', 'wb') as f:
    pickle.dump(clf, f)

#读取模型
with open(f'rf.pickle.dat', 'rb') as f:
    model = pickle.load(f)

from sklearn.externals import joblib
joblib.dump(clf, "rf.pickle.dat")   #保存模型
clf = joblib.load("rf.pickle.dat")  #读取模型

#######################################################
#分组不同的列应用不同的函数, 具体的办法是向agg传入一个从列名映射到函数的字典
df = df.groupby(['hotel_id_m']).agg({'to_hotel_id':'count','score':'mean'})

#######################################################
#差集
ret_list = list(set(a_list)^set(b_list))

#######################################################
# 加特征
# 加上偏度，众数与峰度
# 偏度（Skewness）是描述某变量取值分布对称性的统计量。
    # Skewness=0     分布形态与正态分布偏度相同。
    # Skewness>0     正偏差数值较大，为正偏或右偏。长尾巴拖在右边。
    # Skewness<0     负偏差数值较大，为负偏或左偏。长尾巴拖在左边。
# 峰度（Kurtosis）是描述某变量所有取值分布形态陡缓程度的统计量。 它是和正态分布相比较的。
    # Kurtosis=0     与正态分布的陡缓程度相同。 
    # Kurtosis>0     比正态分布的高峰更加陡峭——尖顶峰
    # Kurtosis<0     比正态分布的高峰来得平缓——平顶峰 
data['skew'] = data.skew()
data['mode'] = data.mode(numeric_only=True).transpose()
data['kurtosis'] = data.kurt(numeric_only=True)

# 计算相关性
round(tmp.corr(method='spearman'),2)
coefficients = ['pearson', 'kendall', 'spearman']
csv_corr = {}
for coefficient in coefficients:
    csv_corr[coefficient] = data.corr(method=coefficient)['rns'].transpose()
    data['rns'].corr(data['month'],method=coefficient)
#######################################################    
# 寻找极小点
for order_dt in order_dts：  # 遍历所有的预定日期
    list = [20,18,5,15,7,5,6,5,4,3,2,1,5]
    list_min = []
    pd.DataFrame(list).plot()
    left = -1  # 用于判断的变量名
    right = 0  # 用于判断的变量名
    for i in range(len(list)-1):  # 按照顺序从第0个变量遍历到倒数第一个
        if i == 0:
            left = -1
        else:
            left = right
        right = list[i+1] - list[i]
        if left < 0 and right > 0:
            list_min.append(i)
            print(i,list[i]) # 这个order_dt下 adv_days = i的这一天是极小值
    for k in range(len(list_min)-1):
        print(pd.DataFrame(list[list_min[k]:list_min[k+1]+1]).kurt())
        pd.DataFrame(list[list_min[k]:list_min[k+1]+1]).plot()

#######################################################
#测试模型
cd E:\jointwisdom\data\rns

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pickle
import os
import re
from datetime import datetime
os.environ['NLS_LANG']='SIMPLIFIED CHINESE_CHINA.UTF8'
from sqlalchemy import create_engine
engine_shop = create_engine('mysql+pymysql://rms_forecast_r2:NTM1MjIxMzU5MDhi@106.75.117.35:13471/forecast')
engine_ctrip  = create_engine('mysql+pymysql://rms_plus_w:hTTkOzQ3tmBlNd8rK@113.209.195.197:33179/rms_plus_monitor')
engine_zone = create_engine('mysql+pymysql://rms_plus_w:hTTkOzQ3tmBlNd8rK@10.201.6.123:3631/forecast')

#特征重要性排名
data = pd.read_csv('A_x.csv.gz')
model =joblib.load('rf.pickle.dat')
a=pd.DataFrame()
a = pd.DataFrame(model.best_estimator_.feature_importances_,columns=['importance'],index = data.columns[1:])
a.sort_values(by='importance',ascending=True).iloc[-8:].plot.barh(figsize=(6,3))
print(a.sort_values(by='importance',ascending=False))

def holiday_tag():
    header_columns = {
        'specialday': ['specialday_nm', 'live_dt', 'tag', 'year', 'tag2', 'day_num'],
    }
    data = {}
    data['specialday'] = pd.read_csv(r"E:\jointwisdom\data\model_test/holiday_new.txt",
                                     sep='#', names=header_columns['specialday'], parse_dates=['live_dt']
                                     )
    data['specialday'] = data['specialday'].drop_duplicates('live_dt',keep='last').drop(['specialday_nm'], axis=1)
    holiday = data['specialday'].set_index(['live_dt']).resample('D')
    holiday = holiday.asfreq(fill_value=0)
    holi = holiday['tag']
    return holi

def get_comp_x(start,num):

    holi = holiday_tag()
    df = holi.reset_index()
    for target in ['day', 'dayofweek', 'month']:
        df[target] = df.live_dt.apply(lambda x: getattr(x, target))
    live_dt_index = pd.date_range(start=start, periods=num)
    df = df.set_index('live_dt').loc[live_dt_index]
    df.index.name = 'live_dt'
    return df

def load_and_predict_comp(start ,num):
    A_x = get_comp_x(start,num)
    A_y = pd.read_csv(f'compA_y.csv.gz', header=[0], index_col=[0])
    pred_A_y_list = []
    for i in range(len(list(A_y.columns))):
        col = list(A_y.columns)[i]
        column = col
        with open(f'comp{column}rf.pickle.dat','rb') as file:
            model = pickle.load(file)
        pred_A_y = pd.DataFrame(model.predict(A_x), columns=[col], index=A_x.index)
        pred_A_y_list.append(pred_A_y)
    pred_A_y_all = pd.concat(pred_A_y_list, axis=1)
    pred = pd.concat([pred_A_y_all])
    return pred
#模型预测值
def load_and_predict(start ,num,htl_cd):
    comp_x =  get_comp_x(start ,num)
    comp_result = load_and_predict_comp(start ,num)
    A_x = pd.concat([comp_x, comp_result], axis=1)
    #竞争酒店数据开关
    # A_x.iloc[:,4:] = 0
    with open(f'rf.pickle.dat','rb') as file:
        model = pickle.load(file)
    pred_A_y = pd.DataFrame(model.predict(A_x).astype(int), columns=['rf_fc'], index=A_x.index)
    # pred_A_y['htl_cd'] = htl_cd
    pred = pd.concat([pred_A_y,A_x],axis=1)
    return pred

#酒店名称
def get_hotel_name(engine_ctrip,htl_tmp):
    sql = 'select htl_cd,hotel_nm name from htl_cd__hotel_nm__map where htl_cd in ({})'.format(htl_tmp)
    name_htl = pd.read_sql(sql,engine_ctrip)
    return name_htl
htl_list = data.htl_cd.unique().astype(str)
name_htl = get_hotel_name(engine_ctrip, ','.join(htl_list))
#线上预测值
def processsmat(engine,start_day,end_day,htl_cd):
    sql = '''select htl_cd,live_dt,occ rns ,rev  from f_forecast_hotel
                 where id in (select Max(id) From f_forecast_hotel 
        where live_dt between {} and {} and htl_cd={}
        Group by live_dt)
        '''.format(start_day,end_day,htl_cd)
    data=pd.read_sql(sql,engine)
    return data

#实际值(店铺上传)
def processshop(engine,start_day,end_day,htl_cd):
    sql = '''select bb.htl_cd htl_cd,DATE_FORMAT(bb.live_dt_inv, '%%Y-%%m-%%d') live_dt,
        bb.observe_dt_inv observe_dt_inv ,bb.rooms_inv inv,bb.occ_otb otb,bb.rev_otb rev,
        bb.adv_days,now() as etl_update_time 
        from(select * from (select a.htl_cd htl_cd,a.live_dt live_dt_inv ,
        a.observe_dt observe_dt_inv ,a.rooms rooms_inv  from (select htl_cd,
        live_dt ,observe_dt,rooms from f_inventory_hotel  
        where id in (select Max(id) From  (select * from f_inventory_hotel 
        where live_dt between {} and {} and htl_cd={}) a 
        Group by live_dt)) as a) as aa 
        left join (select live_dt live_dt_otb ,observe_dt observ_dt_otb,live_dt-observe_dt adv_days,occ occ_otb,rev rev_otb
        from f_order_booking_hotel  
        where id in (select Max(id) 
        From  (select * from f_order_booking_hotel 
        where live_dt between {} and {} and htl_cd={}) a 
        Group by live_dt)) as b on aa.live_dt_inv=b.live_dt_otb) as bb
        '''.format(start_day,end_day,htl_cd,start_day,end_day,htl_cd)
    data=pd.read_sql(sql,engine)
    return data

start_day = '20190115'
end_day = '20190125'
htl_cd = 202697
from datetime import timedelta
t = (pd.to_datetime(end_day) - pd.to_datetime(start_day)).days + 1
pd.to_datetime(start_day) - timedelta(days=30)
live_dt_index = pd.date_range(start=start_day, periods=t)
#df = df.set_index('live_dt').loc[live_dt_index]
data_pred = load_and_predict(start_day,t,htl_cd)
data_mat = processsmat(engine_shop,start_day,end_day,htl_cd)
data_true = processshop(engine_shop,start_day,end_day,htl_cd)

data_mat.index = pd.to_datetime(data_mat.live_dt)
data_mat['true_rns'] = data_true.otb.values
data_mat[['rns','true_rns']].plot()

#######################################################
#测试环境造数据mySQL
from sqlalchemy import create_engine
import pandas as pd
import random
import pymysql
pymysql.install_as_MySQLdb()
def process(htl_cd_i,start_d,days,sub_cd_i):
    data = pd.DataFrame(columns=['htl_cd','sub_cd','live_dt','observe_dt','occ','seg_type','rev'])
    data['live_dt'] = pd.date_range(start_d, periods=days, freq="D")
    data['observe_dt'] = pd.date_range(start_d, periods=days, freq="D")
    data['htl_cd'] =  htl_cd_i
    data['sub_cd'] = sub_cd_i
    data['occ'] = [random.randint(50,200) for _ in range(days)]
    data['seg_type'] = 1
    data['rev'] = data['occ']*200
    return data
htl_cd = 101109
sub_cd_i = 'VIP_A'
start_d = '2018-09-01'
days = 150
table = 'f_order_booking_market'
#######################
data = process(htl_cd,start_d,days,sub_cd_i)
db_host = '192.168.13.156'  # 主机
db_user = 'wisdom'  # 用户名
db_pass = '13JWpgaPal9N1ebE'  # 密码
db_port = '3306'  # 端口号
db_name = 'forecast'  # 数据库名
engine = create_engine("mysql+mysqldb://" + db_user + ":" + db_pass + "@" + db_host + ":" + db_port + "/" + db_name + "?charset=utf8")
pd.io.sql.to_sql(data , table , con=engine, if_exists='append', index=False)

#######################################################
# mySQL取数
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:123456@192.168.19.50:3306/revenue')
sql = "select * from pms_order limit 10"
data=pd.read_sql(sql,engine)

#######################################################
# 判断目录是否存在
import os
dirs = 'E:/jointwisdom/qq'
if not os.path.exists(dirs):
    os.makedirs(dirs)

# 判断文件是否存在
os.path.exists(r'E:\git\RMS_newfeatures\script/model_rns.txt')
import os
 with open('E:/jointwisdom/model_rns1.txt', 'w+') as f:
        f.write('w12' '\n')
        f.write('ww12' '\n')


#######################################################
#条形图显示数值
###条形图
df_wai2.groupby(['day', '地点'])['手机MAC'].unique().map(lambda x: len(x)).unstack().plot(
    kind='bar', figsize=(9, 5)).legend(fontsize=14)
df_wai3 = df_wai2.groupby(['day', '地点'])['手机MAC'].unique().map(lambda x: len(x)).unstack().fillna(0)
df_wai3.columns = ['23栋', '5栋']
x = np.arange(df_wai3.shape[0])
y = df_wai3[['23栋']].values
y1 = df_wai3[['5栋']].values
for a1, b1 in zip(x, y):
    plt.text(a1, b1 + 0.05, '%.0f' % b1, ha='center', va='bottom', fontsize=12)
    # plt.text(a1, b1 + 0.5, '%.1f' % b1, ha='center', va='bottom', fontdict={
    #     'color': 'green' if b1 < -5 else 'red' if b1 > 5 else 'black', 'size': 11})
for a1, b1 in zip(x, y1):
    plt.text(a1, b1 + 0.05, '%.0f' % b1, ha='center', va='bottom', fontsize=12)

plt.xlabel('日期', fontsize=12)
plt.xticks(rotation=360, fontsize=12)
plt.ylabel('拿外卖人数', fontsize=12, rotation=0)
plt.title('每天不同宿舍楼拿外卖人数')
####
bins = [0.2,0.4,0.6,0.8,1]
labels = ["0.2~0.4","0.4~0.6","0.6~0.8","0.8~1"]
corr_df["corr_cut"] = pd.cut(corr_df.corr1, bins, labels=labels)

plt.subplots(figsize = (8, 4))
grouped_values = corr_df.groupby("corr_cut").count().reset_index()
grouped_values.columns = ['corr_cut','htl_count','corr1']
#设置绘制柱状图的颜色
pal = sns.color_palette("Greens_d",len(grouped_values))
rank = grouped_values["htl_count"].argsort().argsort()
g = sns.barplot(x="corr_cut",y="htl_count",data=grouped_values,palette=np.array(pal[::-1])[rank])
#在柱状图的上面显示各个类别的数量
for index,row in grouped_values.iterrows():
    #在柱状图上绘制该类别的数量
    g.text(row.name,row.htl_count,round(row.htl_count,2),color="black",ha="center")
plt.show()
corr_df.corr_cut.value_counts()

#饼图
plt.figure(figsize=(12,6))
plt.subplot(241)
b=data.corr_cut.value_counts()
plt.pie(b,labels=b.index,autopct='%1.2f%%')
plt.title("corr ratio")

#######################################################
#生成动图
import imageio
frames = []
image_list = [str(i)+str(j)+'.png' for i in [2013,2014,2015,2016,2017,2018] for j in ['01','02','03','04','05','06','07','08','09','10','11','12']]
for i in image_list:
    frames.append(imageio.imread(i))
imageio.mimsave('alan2.gif',frames,'GIF',duration = 1) #duration 间隔时间

#######################################################
#数据滑窗法
def create_interval_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i+look_back])
        dataY.append(dataset[i+look_back])
    return np.asarray(dataX), np.asarray(dataY)
df = pd.read_csv("matrix.csv")
dataset = np.asarray(df.iloc[:,1])
dataX, dataY = create_interval_dataset(dataset, 3)

#######################################################
#标准化一列
data['loan_amnt'] = MinMaxScaler(feature_range=(500000,80000000)).fit_transform(
        data['loan_amnt'].values.reshape(-1,1))

#######################################################
#词云图
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#读取文件,返回一个字符串，使用utf-8编码方式读取，该txt文本文件位于此python同以及目录下
#注意：该txt文本文件必须是utf-8编码
f = open(r'D:\agness/keshihua.txt','r',encoding='utf-8').read()
#生成一个词云对象
wordcloud = WordCloud(
        background_color="white", #设置背景为白色，默认为黑色
        width=1500,              #设置图片的宽度
        height=1500,              #设置图片的高度
        margin=10,               #设置图片的边缘
        scale=15
        ).generate(f)
# 绘制图片
plt.imshow(wordcloud)
# 消除坐标轴
plt.axis("off")
wordcloud.to_file("wordcloud1.png")
#######################################################
#学习曲线
'#learning_curve()：这个函数主要是用来判断（可视化）模型是否过拟合的'
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

(X,y) = datasets.load_digits(return_X_y=True)
# print(X[:2,:])
train_sizes,train_score,test_score = learning_curve(
        RandomForestClassifier(),X,y,train_sizes=[0.1,0.2,0.4,0.6,0.8,1],cv=10,scoring='accuracy')
train_error =  1- np.mean(train_score,axis=1)
test_error = 1- np.mean(test_score,axis=1)
plt.plot(train_sizes,train_error,'o-',color = 'r',label = 'training')
plt.plot(train_sizes,test_error,'o-',color = 'g',label = 'testing')
plt.legend(loc='best')
plt.xlabel('traing examples')
plt.ylabel('error')
plt.show()

'#validation_curve()：这个函数主要是用来查看在参数不同的取值下模型的性能 '
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt

(X,y) = datasets.load_digits(return_X_y=True)
# print(X[:2,:])
param_range = [10,20,40,80,160,250]
train_score,test_score = validation_curve(
        RandomForestClassifier(),X,y,param_name='n_estimators',param_range=param_range,cv=10,scoring='accuracy')
train_score =  np.mean(train_score,axis=1)
test_score = np.mean(test_score,axis=1)
plt.plot(param_range,train_score,'o-',color = 'r',label = 'training')
plt.plot(param_range,test_score,'o-',color = 'g',label = 'testing')
plt.legend(loc='best')
plt.xlabel('number of tree')
plt.ylabel('accuracy')
plt.show()

#######################################################
#折线图,不同的值,不同线条类型
a=tmp2[['int_rate_cut','num','pred']].groupby('int_rate_cut').mean()
a.iloc[:,0].plot(figsize=(10,4),linewidth=3,linestyle='-')
a.iloc[:,1].plot(figsize=(10,4),linewidth=3,linestyle='--')
plt.legend(["实际值", "预测值"])
plt.xticks(range(len(a.index)),(a.index))

#######################################################
#读取多个csv文件,并以文件名命名
import pandas as pd
import os
os.chdir(r'E:\data\credit\credict1\data/')
loanfile=os.listdir()
createVar=locals()
for i in loanfile:
    if i.endswith('csv'):
        createVar[i.split('.')[0]]=pd.read_csv(i,encoding='gbk')
        print(i.split('.')[0])

#######################################################
#除去价格为 0 的数据
data.replace(0,np.nan,inplace=True)

#######################################################
#GridSearchCV分类
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))

#######################################################
print(metrics.confusion_matrix(y_test, y_pred_class))
#Markdown 显示混淆矩阵图形
![image.png](attachment:image.png)

#######################################################
import seaborn as sns
sns.lmplot(x='al', y='ri', data=glass, ci=None) #回归线散点图

#######################################################
#分段均值
lower_mileage_price = train[train.miles < miles].price.mean()
higher_mileage_price = train[train.miles >= miles].price.mean()
train['prediction'] = np.where(train.miles < miles, lower_mileage_price, higher_mileage_price)

#######################################################
#根据两列值约束第三列值
temp_table['label'] = temp_table.apply(lambda x:
                                       x['rns'] if (x['rns']>min(x['lower'],x['upper']))&(x['rns']<max(x['lower'],x['upper']))
                                       else x['median'],axis=1)

#######################################################    
#oracle 写入数据
import cx_Oracle
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
net_address = '10.28.110.115:1521/orcl'
engine_bdw = create_engine('oracle://BI_BMK24:BI_BMK24@' + net_address)
try:
    # 上传数据库前先检查当月数据是否存在，存在则删除
    Session = sessionmaker(bind=engine_bmk)
    session = Session()
    session.execute('''
                    DELETE FROM SYM_TBMK_MD_CUST_RANK
                    WHERE data_date <= date'{}' AND
                          data_date >= date'{}' AND
                          rank_type IN ('2')
                    '''.format(end_d, end_m))  # '2019-07-31'   '2019-07-01'
    session.commit()
    session.close()
    recordDf(model_type, '清空当月数据', v_logcontent='success')
except Exception as e:
    recordDf(model_type, '清空当月数据', v_logcontent='fail',
             v_log_adds=str(e).replace("'", " "))
    raise
try:
    cust_rank_df.index = range(len(cust_rank_df))
    num_cust = len(cust_rank_df)
    cust_rank_df = np.array(cust_rank_df).tolist()

    # 上传数据
    # 设置连接数据库参数
    # '10.28.42.22:1521/orcl'切分为['10.28.42.22', '1521', 'orcl']
    param_lst = re.split('[: | /]', net_address)
    ip = param_lst[0]
    port = int(param_lst[1])
    SID = param_lst[2]
    dsn = cx_Oracle.makedsn(ip, port, SID)
    connection = cx_Oracle.connect('BI_BMK24', 'BI_BMK24', dsn)
    cursor = cx_Oracle.Cursor(connection)

    # 批量插入语句
    cursor.prepare(
        'INSERT INTO SYM_TBMK_MD_CUST_RANK(data_date, cust_id, rank_type, dict_val, val) VALUES(:1, :2, :3, :4, :5)')
    cursor.executemany(None, cust_rank_df)
    connection.commit()
    cursor.close()
    connection.close()

    recordDf(model_type, '将数据上传到数据库', v_logcontent='success',
             v_log_adds='流失模型的数据样本量：' + str(num_cust))
except Exception as e:
    recordDf(model_type, '将流失模型的数据上传到数据库', v_logcontent='fail',
             v_log_adds=str(e).replace("'", " "))
    raise

#######################################################
#内存占用
def mem_usage(df):
    return f'{df.memory_usage(deep=True).sum()/1024**2:3.2f} MB'

#批量转categories
def convert_df(df):
    for col in df.columns:
        if df[col].nunique()/df[col].shape[0]<0.5:
            df[col] = df[col].astype("category").cat.codes
    return df

#######################################################
df.nsmallest(5,['rns']) #rns列最小的5行
df.nlargest(5,['rns'])  #rns列最大的5行

#######################################################
import os
import shutil
path = r'E:\Alan\tmp/'
os.remove(path+'tmp.docx')   #删除文件
os.removedirs(path)   #删除空文件夹

shutil.rmtree(path)    #递归删除文件夹(删除非空文件夹)

#######################################################
#选择最新的10条数据
data = data.sort_values(by=['htl_cd','live_dt','fc_dt','update_time'],ascending=[1,1,1,0])
grouped = data.groupby(['htl_cd','live_dt','fc_dt'],as_index=False).head(10)

#######################################################
# 在给定大小的邻域内取中值替代数据值，在邻域中没有元素的位置补0
from  scipy import  signal
signal.medfilt(volume=x, kernel_size=5)
#移动窗口均值
data['rns2'] = data.rns.rolling(window=5,rolling).mean() # 从中间取，前后各取两个值
pandas.rolling_mean(arg, window, min_periods=None, freq=None, center=False, how=None, **kwargs)

#######################################################
# 对训练集和测试集进行相同的缩放
from sklearn.preprocessing import MinMaxScaler
scaler  = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) #X_test_scaled的范围不是0-1,因为用的是训练集的最小值和最大值

#######################################################
# 关闭数据库连接
conn = create_engine('mysql+pymysql:user:passwd@host:port/db?charset=etf-8')
con = conn.connect()
con.close()
conn.dispose()

#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


####################################################### 


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################  


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################    


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################


#######################################################   











