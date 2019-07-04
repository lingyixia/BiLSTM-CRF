# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Temp.py
# @Project       BiLSTM-CRF
# @Product       PyCharm
# @DateTime:     2019-07-02 15:16
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
from pathlib import Path
import random

if __name__ == '__main__':
    data = list()
    with Path('data/data.txt').open() as reader:
        data = reader.readlines()
    random.shuffle(data)
    train = data[:len(data) * 8 // 10]
    dev = data[len(data) * 8 // 10:-len(data) // 10]
    test = data[-len(data) // 10:]
    with Path('data/train.txt').open(mode='w') as writer:
        writer.writelines(train)
    with Path('data/dev.txt').open(mode='w') as writer:
        writer.writelines(dev)
    with Path('data/test.txt').open(mode='w') as writer:
        writer.writelines(test)
