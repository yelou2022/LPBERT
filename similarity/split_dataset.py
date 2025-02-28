# 本文件负责将pair.tsv文件分为训练-验证集，测试集，训练-验证集比例为80%，测试集比例为20%
# 训练-验证集用于模型的五折交叉训练，最后在测试集上测试
import pandas as pd
from sklearn.model_selection import train_test_split


def split_interaction_dataset(file_path, out_dir, test_size=0.3, random_state=42, name=None):
    # 读取数据
    data = pd.read_csv(file_path, sep='\t', header=None, names=['Protein1', 'Protein2', 'Interaction'])

    # 获取所有蛋白质（排序以确保顺序一致）
    all_proteins = set(data['Protein1']).union(set(data['Protein2']))
    sorted_proteins = sorted(all_proteins)  # 排序确保顺序一致

    # 分配训练集和测试集蛋白质
    train_proteins, test_proteins = train_test_split(
        sorted_proteins,  # 使用排序后的蛋白质列表
        test_size=test_size,
        random_state=random_state
    )

    # 确保训练集和测试集蛋白质不重叠
    train_proteins = set(train_proteins)
    test_proteins = set(test_proteins)

    # 分离训练集和测试集
    train_data = data[(data['Protein1'].isin(train_proteins)) & (data['Protein2'].isin(train_proteins))]
    test_data = data[(data['Protein1'].isin(test_proteins)) & (data['Protein2'].isin(test_proteins))]

    # 保存数据到TSV文件
    if name is None:
        train_data.to_csv(f'{out_dir}/train_dataset.tsv', sep='\t', index=False, header=False)
        test_data.to_csv(f'{out_dir}/test_dataset.tsv', sep='\t', index=False, header=False)
    else:
        train_data.to_csv(f'{out_dir}/train_dataset{name}.tsv', sep='\t', index=False, header=False)
        test_data.to_csv(f'{out_dir}/test_dataset{name}.tsv', sep='\t', index=False, header=False)

    print("数据集已分离并保存为 'train_dataset.tsv' 和 'test_dataset.tsv'")


if __name__ == '__main__':
    # 原始数据集
    split_interaction_dataset('原始数据/pair.tsv', '最终数据集')
    # 90%相似性数据集
    split_interaction_dataset('output/pair90.tsv', '最终数据集', name='90')
    # 80%相似性数据集
    split_interaction_dataset('output/pair80.tsv', '最终数据集', name='80')
    # 70%相似性数据集
    split_interaction_dataset('output/pair70.tsv', '最终数据集', name='70')
    # 60%相似性数据集
    split_interaction_dataset('output/pair60.tsv', '最终数据集', name='60')
    # 50%相似性数据集
    split_interaction_dataset('output/pair50.tsv', '最终数据集', name='50')
