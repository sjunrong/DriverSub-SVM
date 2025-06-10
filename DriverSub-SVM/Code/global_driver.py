import numpy as np
import pandas as pd
from tqdm import tqdm
from Code.personalized_driver import BPR_score_matrix_filter, MutMatrix_filter


def condorcetRanking(score_matrix, mutation_matrix):
    # 找到有突变的样本数目，即每个基因中有突变的样本数
    pretrunc = mutation_matrix.sum(axis=0)
    trunc = mutation_matrix.columns[pretrunc > 0]
    trunc = trunc.intersection(score_matrix.columns)
    score_matrix = score_matrix.loc[:, trunc]
    mutation_matrix = mutation_matrix.loc[:, trunc]

    num_genes = len(trunc)
    cmat = np.zeros((num_genes, num_genes))

    def parFun(i, trunc, score_matrix, mutation_matrix, num_genes):
        rowTmp = np.zeros(num_genes)
        colTmp = np.zeros(num_genes)
        votemat = mutation_matrix.values
        fightmat = score_matrix.values

        fightvec = fightmat[:, i]
        fightres = (fightmat - fightvec.reshape(-1, 1)) * -1
        fightres = np.where(fightres > 0, 1, 0)

        testmut = np.where((votemat == 0) & (votemat[:, i].reshape(-1, 1) == 0))

        if len(testmut[0]) > 0:
            fightres[testmut] = 0
            rowTmp = fightres.sum(axis=0)
            fightres = 1 - fightres
            fightres[testmut] = 0
            colTmp = fightres.sum(axis=0)
        else:
            rowTmp = fightres.sum(axis=0)
            fightres = 1 - fightres
            colTmp = fightres.sum(axis=0)

        rowTmp[i] = colTmp[i] = 0
        return rowTmp, colTmp

    for i in tqdm(range(num_genes), desc='Processing genes', unit='gene'):
        rowTmp, colTmp = parFun(i, trunc, score_matrix, mutation_matrix, num_genes)
        cmat[i, :] = rowTmp
        cmat[:, i] = colTmp

    wins = cmat.sum(axis=1)
    losses = cmat.sum(axis=0)
    copelandcriterion = pd.Series(wins / (wins + losses), index=trunc).sort_values(ascending=False)
    return cmat, copelandcriterion


# 调用函数
cmat, copelandcriterion = condorcetRanking(BPR_score_matrix_filter, MutMatrix_filter)
print(copelandcriterion)

# # 取前X个基因并转化为set列表格式
# select_global_genes = set(copelandcriterion.head(300).index)
# 选择分数大于等于0.5的基因并转化为set列表格式
select_global_genes = set(copelandcriterion[copelandcriterion >= 0.5].index)


# 创建为 DataFrame
drivergenes = pd.DataFrame(copelandcriterion[copelandcriterion >= 0.5].index, columns=['Genes'])
# 保存 DataFrame 到 CSV 文件
drivergenes.to_csv('brca_drivergenes.csv', index=False)

print("全局驱动基因识别完成")