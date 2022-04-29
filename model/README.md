### 计算Loss
- AlignLoss: 两个4分类，QA拟合QAR的GT，QAR拟合QA的GT
- AlignLoss16： 一个16分类，拟合4x4个相似度中唯一正确的一个

### 计算相似度
- InnerProduct: 向量内积
- CosineSimilarity: 向量的余弦相似度
  - AddMarginProduct: 计算余弦相似度，GT项减去一个margin。https://zhuanlan.zhihu.com/p/76540469
  - ArcMarginProduct: 非GT项计算余弦相似度，GT项计算向量夹角加上一个margin后的余弦值。https://zhuanlan.zhihu.com/p/76541084

### 损失函数
- CE
- BCE