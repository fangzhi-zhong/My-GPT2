# class 2

## torch's knowledge

* view()


* passage: pytorch internals
https://blog.ezyang.com/2019/05/pytorch-internals/

* 权重的初始化：（针对于激活函数为tanh的一层网络）

  * 将权重初始化为均值为0，标准差为 1 的高斯分布

  * 该层神经网络的输出值，也要符合均值为 0 ，标准差为 1

    * reason： tanh在(-1, 1)之间的梯度更明显，在训练的时候梯度可以更好的向后传播

  * 如何达到上述效果：

    ```
    x = torch.randn(1000,10) # 训练集
    w = torch.randn(10, 100) / 10**0.5 # 该层网络的权重 , 10 是该层网络的输入dim，具体原理可以参考《概率与统计》
    # 这样w的输出值 符合均值为0，标准差为1
    ```

  * 一篇论文具体介绍了如何更好的初始化权重[《Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》](https://arxiv.org/pdf/1502.01852)

  * 论文成果已经集成到torch里面了

    ![image-20250406171814068](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406171814068.png)

    ![image-20250406171921734](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406171921734.png)

* 批归一化，可以视为一种正则化效应，防止过拟合

  * 但是批归一化，和批次这个参数耦合在一起了，这在数学上是一种很不好的现象，但是在神经网络的训练中确实取得了不错的效果
  * 更进一步，为了规避这种不好的现象，研究出了一些其他的归一化技术：层归一化，组归一化等等





## 画图

画直方图

`torch.histogram`

在调参时需要关注的值：

* 前向传播时的Tanh层的输出值

![image-20250406222529103](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406222529103.png)

* 向后传播时，Tanh层的梯度值

![image-20250406222613274](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406222613274.png)

* 向后传播时，W权重值的梯度

![image-20250406230124991](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406230124991.png)

* W权重的更新率，`lr*W.grad/W.data` 最好在$10^{-3}$左右

![image-20250406230108476](C:\Users\26091\AppData\Roaming\Typora\typora-user-images\image-20250406230108476.png)