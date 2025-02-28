# 统计pair文件中阴性样本和阳性样本的数量

def count_pos_neg(pair_file):
    pos_num = 0
    neg_num = 0
    with open(pair_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[2] == '1':
                pos_num += 1
            else:
                neg_num += 1

    print(f'{pair_file}文件：阳性样本：{pos_num}，阴性样本：{neg_num}.')


if __name__ == '__main__':
    pair_file = '../原始数据/pair.tsv'
    count_pos_neg(pair_file)

    pair50 = 'pair50.tsv'
    count_pos_neg(pair50)

    pair60 = 'pair60.tsv'
    count_pos_neg(pair60)

    pair70 = 'pair70.tsv'
    count_pos_neg(pair70)

    pair80 = 'pair80.tsv'
    count_pos_neg(pair80)

    pair90 = 'pair90.tsv'
    count_pos_neg(pair90)