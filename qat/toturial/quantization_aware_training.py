"""
量化感知训练通常包括以下几个步骤：

定义量化器：选择合适的量化器，例如定点量化器或浮点量化器，并设置量化位数和量化误差等参数。
定义损失函数：选择适合的损失函数，例如交叉熵损失函数或均方误差损失函数。
进行前向传播：将输入数据送入深度神经网络，进行前向传播，得到模型的输出。
计算损失函数：将模型的输出与标签数据进行比较，计算损失函数。
进行反向传播：根据损失函数计算梯度，进行反向传播，更新模型的参数。
进行量化：在反向传播时，将梯度进行量化，得到一组离散的梯度值。
更新模型：根据量化后的梯度值更新模型的参数。
重复以上步骤：重复以上步骤，直到模型收敛或达到预定的训练轮数。
————————————————
版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
原文链接：https://blog.csdn.net/qq_44089890/article/details/130021675
"""
"""
    QAT通常使用大约10%的原始训练计划，从初始训练学习率的1%开始，以及余弦退火学习率计划，该计划遵循余弦周期的递减一半，下降到初始微调学习率（初始训练学习速率的0.01%）的1%。

    量化感知训练（本质上是一个离散的数值优化问题）不是数学上解决的问题。根据经验，给出以下一些建议：
    1、为了使 STE 近似效果良好，最好使用较小的学习速率。大的学习速率更可能放大STE近似引入的方差，并破坏训练的网络。

    2、在训练期间不要改变量化表示（Scale尺度），至少不要太频繁。每一步都改变尺度，实际上就像每一步改变数据格式一样，这很容易影响收敛。
"""