def tsv_to_fasta(tsv_file, fasta_file):
    with open(tsv_file, 'r') as tsv, open(fasta_file, 'w') as fasta:
        for line in tsv:
            columns = line.strip().split('\t')
            if len(columns) >= 2:
                header, sequence = columns[0], columns[1]
                fasta.write(f'>{header}\n{sequence}\n')


def fasta2tsv(fasta_file, tsv_file):
    with open(fasta_file, 'r') as fasta, open(tsv_file, 'w') as tsv:
        header = None
        sequence = []
        for line in fasta:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    tsv.write(f'{header}\t{sequence[0]}\n')
                header = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if header:
            tsv.write(f'{header}\t{"".join(sequence)}\n')


if __name__ == '__main__':
    # # 用于将tsv格式的序列数据转换为fasta格式，以便CD-HIT工具的输入
    # tsv = './原始数据/sa_seq.tsv'
    # fasta = './output/sa.fasta'
    #
    # tsv_to_fasta(tsv, fasta)
    # -----------------------------------------------------------
    # 将由CD-HIT处理好的fasta格式的序列数据转换为tsv格式，用于后续的数据处理
    fasta_50 = './Processed_by_CD-HIT/output_0.5.fasta'
    fasta_60 = './Processed_by_CD-HIT/output_0.6.fasta'
    fasta_70 = './Processed_by_CD-HIT/output_0.7.fasta'
    fasta_80 = './Processed_by_CD-HIT/output_0.8.fasta'
    fasta_90 = './Processed_by_CD-HIT/output_0.9.fasta'

    tsv_50 = './output/seq50.tsv'
    tsv_60 = './output/seq60.tsv'
    tsv_70 = './output/seq70.tsv'
    tsv_80 = './output/seq80.tsv'
    tsv_90 = './output/seq90.tsv'

    fasta2tsv(fasta_50, tsv_50)
    fasta2tsv(fasta_60, tsv_60)
    fasta2tsv(fasta_70, tsv_70)
    fasta2tsv(fasta_80, tsv_80)
    fasta2tsv(fasta_90, tsv_90)
