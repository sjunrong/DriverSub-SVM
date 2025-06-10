import time
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import  GridSearchCV, cross_val_predict, StratifiedKFold
from Code.Random_walk import result, start_time
from sklearn.svm import SVC
from Code.personalized_driver import results
from Code.global_driver import select_global_genes

# 分离样本和标签
X = results.iloc[:, 1:-1].values
Y = results.iloc[:, -1:].values

np.random.seed(2 ** 32-1)

# 获取results的列名
gene_columns = results.columns[1:-1]
# 找到select_global_genes在gene_columns中的列索引
selected_indices = [gene_columns.get_loc(gene) for gene in select_global_genes if gene in gene_columns]
# 筛选X，只保留在select_global_genes中的列
X = X[:, selected_indices]


# # 确保select_global_genes中的基因存在于gene_columns中
# selected_genes = [gene for gene in select_global_genes if gene in gene_columns]
# # 筛选出包含selected_genes的新DataFrame
# filtered_results = results[['pan.samplesID'] + selected_genes + ['Subtype']]
# # 将新生成的结果保存为CSV文件
# filtered_results.to_csv('brca_shapdata.csv', index=False)

# # no driver genes实验，获取基因列的名字
# gene_columns = result.columns[1:-1]
# # 找到不在 select_global_genes 中的基因名称
# non_selected_genes = [gene for gene in gene_columns if gene not in select_global_genes]
# # 获取不在 select_global_genes 中的基因的列索引
# non_selected_indices = [gene_columns.get_loc(gene) for gene in non_selected_genes]
# # 筛选 X，只保留不在 select_global_genes 中的列
# X = X[:, non_selected_indices]


# 使用SMOTE进行过采样
smote = SMOTE()
X, Y = smote.fit_resample(X, Y.ravel())


# # 重置随机数生成器的状态
# np.random.seed(None)
# # 打乱数据顺序
# num_samples, num_features = X.shape
# shuffle_index = np.random.permutation(num_samples)
# X = X[shuffle_index, :]
# Y = Y[shuffle_index]

end_time = time.time()
duration = end_time - start_time
print("数据处理时间：", duration, "秒")


# 定义模型并进行参数搜索
params = {'kernel': ['rbf'],
          'C': np.logspace(-1, 5, 10),
          'gamma': np.logspace(-4, 2, 10)}
svm_model = SVC(decision_function_shape='ovo')
grid_search = GridSearchCV(svm_model, param_grid=params, cv=10, n_jobs=1)
grid_search.fit(X, Y)
# 使用最优参数的模型
ovo_classif = SVC(decision_function_shape='ovo', **grid_search.best_params_)

# # 获取所有超参数组合及其对应的准确率结果
# svm_results = grid_search.cv_results_
# params_and_accuracies = list(zip(svm_results['params'], svm_results['mean_test_score']))
# # 输出每个超参数组合及其对应的accuracy
# for params, accuracy in params_and_accuracies:
#     print(f"Params: {params}, Accuracy: {accuracy:.4f}")

# best_params = {'kernel': 'rbf', 'C': 46.41588833612777, 'gamma': 0.046415888336127774}
# # 定义模型并使用最优参数
# ovo_classif = SVC(decision_function_shape='ovo', **best_params)

# 使用交叉验证预测Y值
Y_pred = cross_val_predict(ovo_classif, X, Y.ravel(), cv=10)


end_time1 = time.time()
duration1 = end_time1 - end_time
print("模型训练和测试时间：", duration1, "秒")
print("stop")