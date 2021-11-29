from os import write
from mylib.utils import (
    write_csv, load_csv
)

STSB_PATHS = [
    './datasets/Stsbenchmark/sts-train.csv',
    './datasets/Stsbenchmark/sts-dev.csv',
    './datasets/Stsbenchmark/sts-test.csv'
]

def combine_sts_benchmark():
    """
    combine sts-dev, sts-train, sts-test into sts-all file, only reserve 5, 6 columns
    """
    datas = []
    for path in STSB_PATHS:
        with open(path, 'r',  encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                datas.append('\t'.join(line.strip().split('\t')[5:7]) + '\n')
                
    with open('./datasets/Stsbenchmark/sts-all.csv', 'w', encoding='utf-8') as f:
        f.writelines(datas)
    #write_csv(datas, './datasets/Stsbenchmark/sts-all.csv')

combine_sts_benchmark()