# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         Test.py
# @Project       MyEL2019
# @Product       PyCharm
# @DateTime:     2019-06-18 13:23
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
from pathlib import Path
from tf_metrics import f1

def transformData(inputFile):
    sentences = list()
    tags = list()
    with open(inputFile, encoding='utf-8') as reader:
        sentence = list()
        tag = list()
        for line in reader:
            line = line.replace('\n', '')
            if line == '':
                sentences.append(''.join(sentence.copy())+'\n')
                tags.append(''.join(tag.copy())+'\n')
                sentence.clear()
                tag.clear()
            else:
                thisWord = line.split(' ')[0]
                thisTag = line.split(' ')[1]
                sentence.append(thisWord)
                tag.append(thisTag)
    with Path(inputFile.parent.joinpath(inputFile.name.split('.')[0] + '_sentences.txt')).open(mode='w') as writer:
        writer.writelines(sentences)
    with Path(inputFile.parent.joinpath(inputFile.name.split('.')[0] + '_tags.txt')).open(mode='w') as writer:
        writer.writelines(tags)


if __name__ == '__main__':
    for file in Path('data').iterdir():
        transformData(file)
