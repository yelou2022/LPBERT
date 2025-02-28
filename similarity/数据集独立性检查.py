# 定义一个函数，用于判断两个相互作用文件中蛋白质数据是否有交叉
import pandas as pd


def check_independence(file1, file2):
    # 使用pandas读取File的前两列数据，构建蛋白质ID集合
    file1_data = pd.read_csv(file1, sep='\t', header=None, names=['Protein1', 'Protein2', 'Interaction'])
    file2_data = pd.read_csv(file2, sep='\t', header=None, names=['Protein1', 'Protein2', 'Interaction'])

    # 构建集合
    file1_proteins = set(file1_data['Protein1']).union(set(file1_data['Protein2']))
    file2_proteins = set(file2_data['Protein1']).union(set(file2_data['Protein2']))
    # print(f'{len(file1_proteins)}: {file1_proteins}')
    # print(f'{len(file2_proteins)}: {file2_proteins}')]

    # 判断两个集合是否有交集
    overlap = file1_proteins.intersection(file2_proteins)
    if len(overlap) > 0:
        print(f'两个数据集有交集: {overlap}')
    else:
        print('两个数据集独立')


if __name__ == '__main__':
    check_independence('最终数据集/test_dataset.tsv', '最终数据集/train_dataset.tsv')
    check_independence('最终数据集/test_dataset90.tsv', '最终数据集/train_dataset90.tsv')
    check_independence('最终数据集/test_dataset80.tsv', '最终数据集/train_dataset80.tsv')
    check_independence('最终数据集/test_dataset70.tsv', '最终数据集/train_dataset70.tsv')
    check_independence('最终数据集/test_dataset60.tsv', '最终数据集/train_dataset60.tsv')
    check_independence('最终数据集/test_dataset50.tsv', '最终数据集/train_dataset50.tsv')

