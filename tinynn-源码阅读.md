[TOC]

# Tinynn源码阅读笔记

## Core

### Tensor

+ 能否把dist/tuple转为tensor？
+ 测试一下各种运算符，编写测试文件
+ @property装饰器

### ops

+ handle_broadcasting

+ 补充函数文档

+ numpy.clip

  + ```py	
    #整流
    x=np.array([[1,2,3,5,6,7,8,9],[1,2,3,5,6,7,8,9]])
    np.clip(x,3,8)
    
    Out[90]:
    array([[3, 3, 3, 5, 6, 7, 8, 8],
           [3, 3, 3, 5, 6, 7, 8, 8]])
    ```

+ numpy.ravel

[NumPy 中 ravel() 正确打开方式](https://blog.csdn.net/yangjjuan/article/details/103690716)

+ numpy.pad

[numpy.pad()函数使用详解](https://blog.csdn.net/OuDiShenmiss/article/details/105618200)

- [ ] pad的梯度计算