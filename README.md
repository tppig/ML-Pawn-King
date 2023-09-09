# ML-Pawn-King

解决机器学习入门问题：

## 文件目录
- krkopt.data: 原数据集
- krkopt_train.data: 训练数据集，由原数据集分出5000个
- krkopt_test.data: 测试数据集，由原数据集分出剩余的
- krkopt.model: 最终模型
- dataset_device.py: 用于读取原数据集并分出训练集和测试集
- optimal_hyper_param.py: 用于寻找最优超参数并且训练出最终模型`krkopt.model`
- test_model.py: 读取模型并在测试集上进行测试
