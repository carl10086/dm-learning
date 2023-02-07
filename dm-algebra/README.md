## 项目说明:

线性代数 , 微积分, 神经网络 `python` 系列 .

- 原书: `用Python学透线性代数和微积分`
- `!` 开头代表规划中 .暂无
- currently you need python `3.10.x` .

> 如果你的电脑是 m1, 大概率要自己处理下自己的环境问题. openblas 在 m1 上是有点痛点的

```bash
pip install cython pybind11 pythran
pip install numpy
brew info openblas

# 以下可能需要
brew install lapack
```

- 根据 `brew info openblas` 的提示进行环境变量的定制 .

**B系列: Python基础**
 
- b4: [matplotlib 基础画图](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/b4.ipynb)

**C系列: 线性代数**

- c1->c5: 后续补充. 对向量矩阵的基本理解. 
- c6: [高维泛化向量空间](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/c6.ipynb)
- c7: [线性方程组求解和矩阵变换](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/c7.ipynb)

**D系列: 微积分和基础物理**

- d1: [导数和积分](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d1.ipynb)
- d2: [欧拉方法-模拟运动的对象](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d2.ipynb)
- d3: [符号表达式和精确微积分](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d3.ipynb)
- d4: [梯度连接力场和势能场](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d4.ipynb)
- d5: [物理系统和梯度下降](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d5.ipynb)
- d6: [!傅里叶变换模拟声波]()

**E系列: 机器学习基础玩法**

- e1: [函数拟合](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d1.ipynb)
- e2: [logistic 回归和数据分类](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d2.ipynb)
- e3: [手写神经网络-MLP 多层感知机](https://github.com/carl10086/dm-learning/blob/master/dm-algebra/chapters/d2.ipynb)


