# 构建基于社区发现的推荐算法
对比IVFFlat和HNSW两种算法
1. 使用k-means初始化IVFFlat索引
2. HNSW根据近邻关系，可以使用图聚类算法Louvain算法来划分社区。
3. 对比两种算法在不同数据集上的运行时间和内存占用
4. 分析两种算法的优缺点
5. 对比两种算法的acc、recall、f1-score
