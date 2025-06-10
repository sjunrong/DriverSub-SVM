import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.special import expit  # Sigmoid function
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, average_precision_score, ndcg_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Code.Data import symbols, df2, nondrivers, MutMatrix, kegg_drivers
from Code.Random_walk import new_M1
import optuna
import random


class BPR:
    def __init__(self, score_matrix, mut_matrix, symbols, nondrivers, latent_dim=20, learning_rate=0.01, regularization=0.01, num_epochs=20, batch_size=500, random_state=42):
        self.score_matrix = csr_matrix(score_matrix)
        self.mut_matrix = mut_matrix
        self.num_samples, self.num_genes = self.score_matrix.shape
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # 随机初始化样本和基因的隐向量
        np.random.seed(self.random_state)  # 设置随机种子
        # self.sample_factors = np.random.normal(scale=1./latent_dim, size=(self.num_samples, latent_dim))
        # self.gene_factors = np.random.normal(scale=1./latent_dim, size=(self.num_genes, latent_dim))

        # 使用SVD分解score_matrix来初始化隐向量
        self.sample_factors, _, self.gene_factors = svds(self.score_matrix, k=self.latent_dim)
        self.gene_factors = self.gene_factors.T

        # 映射symbols到new_M1的列名索引
        self.symbol_indices = [list(new_M1.columns).index(symbol) for symbol in symbols if symbol in new_M1.columns]
        # 映射nondriver到new_M1的列名索引
        self.nondriver_indices = [list(new_M1.columns).index(nondriver) for nondriver in nondrivers if nondriver in new_M1.columns]

        # 生成训练样本
        self.train_samples = self.generate_samples()

    def generate_samples(self):
        samples = []
        for i in range(self.num_samples):
            # 正样本为在NCG驱动基因列表中且发生突变的基因
            mut_indices = set(np.where(self.mut_matrix[i] == 1)[0])  # 获取突变基因的索引
            positive_indices = [idx for idx in self.symbol_indices if idx in mut_indices]
            # 负样本为 nondriver 中但不在突变基因集合中的基因
            negative_indices = [idx for idx in self.nondriver_indices if idx not in mut_indices]
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            for j in positive_indices:
                for _ in range(1):
                    k = np.random.choice(negative_indices)
                    samples.append((i, j, k))
        return samples

    def bpr_update(self, i, j, k):
        x_uij = np.dot(self.sample_factors[i], self.gene_factors[j] - self.gene_factors[k])
        sigmoid = expit(x_uij)
        grad_u = (self.gene_factors[j] - self.gene_factors[k]) * (1 - sigmoid) - self.regularization * self.sample_factors[i]
        grad_j = self.sample_factors[i] * (1 - sigmoid) - self.regularization * self.gene_factors[j]
        grad_k = -self.sample_factors[i] * (1 - sigmoid) - self.regularization * self.gene_factors[k]

        self.sample_factors[i] += self.learning_rate * grad_u
        self.gene_factors[j] += self.learning_rate * grad_j
        self.gene_factors[k] += self.learning_rate * grad_k

    def train(self):
        for epoch in range(self.num_epochs):
            np.random.shuffle(self.train_samples)
            for start in tqdm(range(0, len(self.train_samples), self.batch_size), desc=f'Epoch {epoch + 1}/{self.num_epochs}'):
                batch = self.train_samples[start:start + self.batch_size]
                for i, j, k in batch:
                    self.bpr_update(i, j, k)
            print(f"Epoch {epoch + 1}/{self.num_epochs} completed.")

    def predict(self):
        return np.dot(self.sample_factors, self.gene_factors.T)



# def objective(trial):
#     latent_dim = trial.suggest_int('latent_dim', 10, 500)
#     learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
#     regularization = trial.suggest_float('regularization', 1e-4, 1e-1, log=True)
#     num_epochs = trial.suggest_int('num_epochs', 10, 100)
#     batch_size = trial.suggest_int('batch_size', 100, 1000)
#
#     score_matrix = new_M1.values
#     bpr_model = BPR(score_matrix, MutMatrix.values, symbols, nondrivers, latent_dim=latent_dim, learning_rate=learning_rate,
#                     regularization=regularization, num_epochs=num_epochs, batch_size=batch_size, random_state=42)
#     bpr_model.train()
#     predict_matrix = bpr_model.predict()
#
#     train_data, test_data = train_test_split(score_matrix, test_size=0.2, random_state=42)
#     train_preds = predict_matrix[:train_data.shape[0], :train_data.shape[1]]
#     test_preds = predict_matrix[train_data.shape[0]:, :test_data.shape[1]]
#
#     # 计算RMSE作为评价指标
#     train_rmse = mean_squared_error(train_data[train_data > 0], train_preds[train_data > 0], squared=False)
#     test_rmse = mean_squared_error(test_data[test_data > 0], test_preds[test_data > 0], squared=False)
#
#     return test_rmse
#
#     # # 计算MAP和NDCG作为评价指标, 将真实值和预测值转化为适合计算MAP和NDCG的格式
#     # y_true = (test_data > 0).astype(int)
#     # y_pred = test_preds
#     # # 计算MAP
#     # map_score = average_precision_score(y_true.flatten(), y_pred.flatten())
#     # return -map_score  # 因为Optuna默认最小化目标，所以这里取负值
#     # # 计算NDCG
#     # ndcg = ndcg_score([y_true.flatten()], [y_pred.flatten()])
#     # return -ndcg
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)
#
# print('Best parameters: ', study.best_params)
# print('Best RMSE: ', study.best_value)
#
# # 使用最佳参数训练BPR模型
# best_params = study.best_params
# 使用最佳参数训练BPR模型
best_params = {
    'latent_dim': 197,
    'learning_rate': 0.0007174445582728159,
    'regularization': 0.0005673009188783027,
    'num_epochs': 36,
    'batch_size': 489
}
# 将数据传递给BPR模型
bpr_model = BPR(new_M1.values, MutMatrix.values, symbols, nondrivers,
                latent_dim=best_params['latent_dim'],
                learning_rate=best_params['learning_rate'],
                regularization=best_params['regularization'],
                num_epochs=best_params['num_epochs'],
                batch_size=best_params['batch_size'])
bpr_model.train()
predict_matrix = bpr_model.predict()

# 生成每个样本的基因排序
def generate_gene_ranking(predict_matrix):
    rankings = {}
    num_samples = predict_matrix.shape[0]
    for i in range(num_samples):
        scores = predict_matrix[i]
        ranking = np.argsort(-scores)
        rankings[i] = ranking
    return rankings

rankings = generate_gene_ranking(predict_matrix)

symbols_set = set(symbols)
kegg_drivers_set = set(kegg_drivers)
# 取每个样本排名前100的基因，并取并集
top_genes_set = set()
sample_top_genes = {}
sample_driver_intersection = {}

for i in range(new_M1.shape[0]):
    top_genes = rankings[i][:40]
    top_genes_set.update(top_genes)
    gene_names = [new_M1.columns[gene_index] for gene_index in top_genes]
    sample_top_genes[i] = gene_names

    # 计算交集
    intersection_i = set(gene_names) & kegg_drivers_set
    sample_driver_intersection[i] = intersection_i

# 将基因编号转化为基因名称
top_genes_list = list(top_genes_set)
top_genes_columns = [new_M1.columns[gene_index] for gene_index in top_genes_list]

# 输出每个样本排名前100的基因与 driver 基因的交集及其个数
for i in range(new_M1.shape[0]):
    intersection_genes = sample_driver_intersection[i]
    print(f"Sample {new_M1.index[i]}: Intersection Genes: {list(intersection_genes)}")
    print(f"Number of Intersection Genes: {len(intersection_genes)}")

# 查看筛选出来的整体基因集与NCG中的驱动基因的交集情况
top_genes_set = set(top_genes_columns)
intersection = kegg_drivers_set.intersection(top_genes_set)
print(f"Number of driver genes: {len(intersection)}")
print("筛选出来的基因与NCG中的驱动基因的交集:")
print(intersection)

# 将 predict_matrix 转化为 DataFrame，并设置行名和列名
BPR_score_matrix = pd.DataFrame(predict_matrix, index=new_M1.index, columns=new_M1.columns)
BPR_score_matrix_filter = BPR_score_matrix[top_genes_columns]
MutMatrix_filter = MutMatrix[top_genes_columns]

# 对随机游走分数矩阵进行filter
# 筛选 new_M1 中的列，只保留并集中存在的基因列
first_filter = new_M1[top_genes_columns]
# 对两个数据框的第一列进行交集操作, 获取交集数据
common_idx = pd.Index(first_filter.index).intersection(pd.Index(df2.iloc[:, 0]))
df1_common = first_filter.loc[first_filter.index.isin(common_idx)]
df1_common.insert(0, 'pan.samplesID', df1_common.index)
df1_common = df1_common.reset_index(drop=True)
df2_common = df2.loc[df2.iloc[:, 0].isin(common_idx)]
df2_common = df2_common.rename(columns={'Unnamed: 0': 'pan.samplesID'})
# 将两个交集数据框拼接起来生成一个新的数据框
results = pd.merge(df1_common, df2_common, on='pan.samplesID', how='inner')
results.to_csv('brca_results.csv', index=False)


# 得到BPR的分数矩阵
# 对两个数据框的第一列进行交集操作, 获取交集数据
common_idx = pd.Index(BPR_score_matrix_filter.index).intersection(pd.Index(df2.iloc[:, 0]))
df1_common = BPR_score_matrix_filter.loc[BPR_score_matrix_filter.index.isin(common_idx)]
df1_common.insert(0, 'pan.samplesID', df1_common.index)
df1_common = df1_common.reset_index(drop=True)
df2_common = df2.loc[df2.iloc[:, 0].isin(common_idx)]
df2_common = df2_common.rename(columns={'Unnamed: 0': 'pan.samplesID'})
# 将两个交集数据框拼接起来生成一个新的数据框
BPR_results = pd.merge(df1_common, df2_common, on='pan.samplesID', how='inner')

print("个性化驱动基因识别结束")