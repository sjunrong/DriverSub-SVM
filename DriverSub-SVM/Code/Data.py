import os

import numpy as np
import pandas as pd

data_dir = "../Data"
# 获取文件InfluenceGraph
influenceGrap_path = os.path.join(data_dir, "BRCA InfluenceGraph.csv")
influenceGraph = pd.read_csv(influenceGrap_path, index_col=0)
Influ_index = influenceGraph.index
Influ_columns = influenceGraph.columns
# 获取文件MutMatrix
MutMatrix_path = os.path.join(data_dir, "BRCA CNV_MutMatrix.csv")
MutMatrix = pd.read_csv(MutMatrix_path, index_col=0)
MutMatrix = MutMatrix.reindex(columns=Influ_index)
# 获取文件OutMatrix
OutMatrix_path = os.path.join(data_dir, "BRCA OutMatrix.csv")
OutMatrix = pd.read_csv(OutMatrix_path, index_col=0)
OutMatrix = OutMatrix.reindex(columns=Influ_columns)

df2_path = os.path.join(data_dir, "BRCA_Subtype_PAM50.csv")
df2 = pd.read_csv(df2_path, header=0)
df2 = df2.rename(columns={"histological_type": "Subtype"})
df2 = df2.dropna(subset=["Subtype"])
# 对样本名去重
df2 = df2.drop_duplicates(subset=["Unnamed: 0"])
# 重设index
df2 = df2.reset_index(drop=True)

# 创建字典将类别映射为数字
mapping = {'Normal': 1, 'LumA': 2, 'Her2': 3, 'LumB': 4, 'Basal': 5}
# 将第二列数据进行映射转换
df2['Subtype'] = df2['Subtype'].map(mapping)

# 读入表格
df_path = os.path.join(data_dir, 'NCG_cancerdrivers_annotation_supporting_evidence.tsv.csv')
df = pd.read_csv(df_path, sep=',')

# df3= df[df['type'] == 'Pan-cancer']
# df1 = df[(df['cancer_type'] == 'breast_cancer') | (df['cancer_type'] == 'pan-cancer_adult') | (df['cancer_type'] == 'pan-cancer_paediatric')]
# # 取出symbol列中的数据，并去重
# symbols1 = df1['symbol'].unique()
# symbols2 = df3['symbol'].unique()
# symbols = np.unique(np.concatenate((symbols1, symbols2)))

df1 = df[(df['cancer_type'] == 'breast_cancer') | (df['primary_site'] == 'multiple')]
# df1 = df[(df['cancer_type'] == 'anaplastic_thyroid_carcinoma') | (df['cancer_type'] == 'papillary_thyroid_cancer') | (df['cancer_type'] == 'parathyroid_carcinoma') | (df['primary_site'] == 'multiple')]
# df1 = df[(df['cancer_type'] == 'gastric_adenocarcinoma') | (df['cancer_type'] == 'pan-gastric') | (df['cancer_type'] == 'gastric_cancer') | (df['cancer_type'] == 'diffuse_gastric_adenocarcinoma') | (df['cancer_type'] == 'mucinous_gastric_cancer') | (df['primary_site'] == 'multiple')]
# 取出symbol列中的数据，并去重
symbols = df1['symbol'].unique()

kegg_driver = df[(df['cancer_type'] == 'breast_cancer')]
kegg_drivers = kegg_driver['symbol'].unique()

# 读取 nondriver.txt 文件，将基因名存入nondriver列表
nondriver_path = os.path.join(data_dir, 'nondriver.txt')
nondriver_df = pd.read_csv(nondriver_path, header=None)
nondrivers = nondriver_df[0].tolist()  # 将第一列转换为列表

print("数据准备结束")