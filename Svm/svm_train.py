# coding:UTF-8

import numpy as np
import sys
sys.path.append(r'D:\studyPythonMl\study_ML_Python3.x\Svm')
import svm


def load_data_libsvm(data_file):
    '''导入训练数据
    input:  data_file(string):训练数据所在文件
    output: data(mat):训练样本的特征
            label(mat):训练样本的标签
    '''
    data = []
    label = []
    f = open(data_file)
    for line in f.readlines():
        cols = line.strip().split(' ')  # 分成多个列单元
        
        # 提取得出label
        label.append(float(cols[0]))  # 第一列是真实值
        # 提取出特征，并将其放入到矩阵中
        index = 0  # 数据特征索引 索引号从0开始
        tmp = []  # 每行的数据，每行样本数据
        for i in range(1, len(lines)):
            items = cols[i].strip().split(":")  # 每列被分成两项
            # 由于特征索引是从0开始的，所以要减去1
            if int(items[0]) - 1 == index:
                tmp.append(float(li[1]))
            else:
                # 如果当前索引号没有对应的特征值，则添加0值；前提是存储的数据已经按照索引号排序
                while(int(items[0]) - 1 > index):
                    tmp.append(0)  # 添加0值进行占位
                    index += 1
                tmp.append(float(items[1]))
            index += 1
        # 如果特征值长度不够，则补充0值保证数据长度一致
        while len(tmp) < 13:
            tmp.append(0)
        data.append(tmp)  # 将完整的数据加入到data变量
    f.close()
    return np.mat(data), np.mat(label).T  # 根据列表创建矩阵

if __name__ == "__main__":
    # 1、导入训练数据
    print ("------------ 1、load data --------------")
    dataSet, labels = load_data_libsvm("heart_scale")
    # 2、训练SVM模型
    print ("------------ 2、training ---------------")
    C = 0.6
    toler = 0.001
    maxIter = 500
    svm_model = svm.SVM_training(dataSet, labels, C, toler, maxIter)
    # 3、计算训练的准确性
    print ("------------ 3、cal accuracy --------------")
    accuracy = svm.cal_accuracy(svm_model, dataSet, labels)  
    print ("The training accuracy is: %.3f%%" % (accuracy * 100))
    # 4、保存最终的SVM模型
    print ("------------ 4、save model ----------------")
    svm.save_svm_model(svm_model, "model_file")
