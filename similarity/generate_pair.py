# 本文件用于从原始相互作用文件中生成pair文件
# 经过CD-HIT处理后，seq文件中的序列已经被去冗余，但是相互作用文件中的序列还是冗余的
# 所以需要将相互作用文件中的序列去冗余，然后生成pair文件

def read_seq(seq_file):
    # 将序列tsv文件中涉及的蛋白质id保存到集合中，用于后续判断
    seq_set = set()  # 使用集合，以免有重复的序列
    with open(seq_file, 'r') as f:
        for line in f:
            protein, seq = line.strip().split('\t')
            seq_set.add(protein)

    print("--------------------------------------")
    print(f'{seq_file}文件中共包含{len(seq_set)}个蛋白质')
    return seq_set


def generate_pair(seq_set, interact_file, output_file):
    # 生成pair文件
    with open(interact_file, 'r') as f:
        data = []  # 存储可用的相互作用数据
        for line in f:
            protein1, protein2, _ = line.strip().split('\t')  # 分别代表两个蛋白质的id，第三个是相互作用标签，在这里不重要
            if protein1 in seq_set and protein2 in seq_set:
                data.append(line)

    # 生成pair文件，将data中的数据写入pair文件
    with open(output_file, 'w') as output:
        for line in data:
            output.write(line)


if __name__ == '__main__':
    seq50 = './output/seq50.tsv'
    seq60 = './output/seq60.tsv'
    seq70 = './output/seq70.tsv'
    seq80 = './output/seq80.tsv'
    seq90 = './output/seq90.tsv'

    seq_set50 = read_seq(seq50)
    seq_set60 = read_seq(seq60)
    seq_set70 = read_seq(seq70)
    seq_set80 = read_seq(seq80)
    seq_set90 = read_seq(seq90)

    print("--------------------------------------\n")
    # 执行下一步操作，生成pair文件

    generate_pair(seq_set50, './原始数据/pair.tsv', './output/pair50.tsv')
    generate_pair(seq_set60, './原始数据/pair.tsv', './output/pair60.tsv')
    generate_pair(seq_set70, './原始数据/pair.tsv', './output/pair70.tsv')
    generate_pair(seq_set80, './原始数据/pair.tsv', './output/pair80.tsv')
    generate_pair(seq_set90, './原始数据/pair.tsv', './output/pair90.tsv')
