方面级情感分析

[2019 搜狐校园算法大赛](https://mp.weixin.qq.com/s?__biz=MzU1Nzc1NjI0Nw==&mid=2247484063&idx=1&sn=96a408be480440210f25964e117cf88a&chksm=fc31b8a7cb4631b1f82b5a67603fc189dc62f437b2bd49488c047746b000fc233663f951d936&mpshare=1&scene=23&srcid=0521APy7uWyCyXnaIkRd5VJQ&sharer_sharetime=1590039258697&sharer_shareid=8de0cec7fe154f6f262a3debda1397d5%23rd)

[2019之江杯————电商评论观点挖掘 亚军方案](https://mp.weixin.qq.com/s?__biz=MzI2NDk4NDEwMQ==&mid=2247483699&idx=1&sn=570499d532ad6fe11c8c1c2ee800db29&chksm=eaa50279ddd28b6f60227a7c5311d4251a29d96e17c792028986f2d14387a31954c3fee2bd04&mpshare=1&scene=23&srcid=0521O63fq0vcJFzsIQObkwwG&sharer_sharetime=1590039824478&sharer_shareid=8de0cec7fe154f6f262a3debda1397d5%23rd)

比赛思路：

1.采用pipeline的方式，将这个任务拆为两个子任务，先预测aspect，根据aspect预测情感极性（ABSA），这两个子任务都可以使用深度学习模型解决。

2.aspect预测采用指针标注的方式解决，标注aspect的头和尾，思路参考苏神在百度信息抽取的baseline。

3.基于aspect的情感分析是一个多分类问题，首先根据分隔符将文本拆成多段，然后拼接aspect出现过的文本，再进行三分类。
