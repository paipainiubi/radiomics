!/usr/bin/env python
# coding: utf-8
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns  #用于绘制热图的工具包
from scipy.cluster import hierarchy  #用于进行层次聚类，话层次聚类图的工具包
from scipy import cluster
import matplotlib
import matplotlib.pyplot as plt

'''
说明：（用小崔改好的代码跑更方便，画出来的热图对应横坐标的0就是第一列特征）
特征表格第一列是label，第二行开始是特征列。
第一行是特征名

可修改的地方：
标准化还是归一化？
聚类方式，参数
'''
out_path = r'D:\DESK\ding'
# path = r'C:\Users\Administrator\Desktop\新建文件夹 20200717\画聚类图 20200720\最终筛选特征82个患者9个特征 07-24（带label）.xlsx'
# 表格里需要有label这个单词
path = r"D:\DESK\ding\bc_lasso_trnshaixuan.xlsx"
df = pd.read_excel(path)  #index_col=0指定数据中第一列是类别名称
label = df['label']
del df['label']

scaler = MinMaxScaler()
df2 = scaler.fit_transform(df)
Z = hierarchy.linkage(df2, method ='ward',metric='euclidean')
hierarchy.dendrogram(Z)#,labels = df2.index)  # 树状图
#plt.savefig('{}/树状图1.tiff'.format(out_path), dpi=600)


lut=dict(zip(label.unique(),"brg"))
row_colors=label.map(lut)
row_colors = row_colors.tolist() # row_colors 是为了画出左侧的label
g = sns.clustermap(df2,method ='ward',metric='euclidean', fmt="d",cmap='RdYlBu_r', row_colors= row_colors) # 聚类图/热图
# sns.clustermap(df2,method ='ward',metric='euclidean', fmt="d", row_colors= row_colors) # 聚类图/热图
#cmap设置颜色。fmt格式设置2format(out_path), dpi=300)

plt.savefig('{}/bm-train.tiff'.format(out_path), dpi=600)

plt.show()
# g.savefig('clustermap.png')

# help(sns.clustermap)
