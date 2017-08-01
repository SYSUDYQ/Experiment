# -*- coding:utf8 -*- 

import os
import numpy as np

path = 'E:\\github\\facebook'
# 文件所包含的中心节点
centreNodes = []
# 最后生成的邻接矩阵 每个元素表示距离
adj_node = np.zeros((4039, 4039))
# 这是一个list list中的每一个元素是一个dict
# 按照字典序存储 每一个网络图对应的featname
featname = []

def getFilesName(path):
	#获取所有一级目录下的文件名
	allFilesName = []
	files = os.listdir(path)
	for f in files:
		if(os.path.isfile(path + '\\' + f)):
			allFilesName.append(f)
	return allFilesName	

def printFilesName(allFilesName):
	for f in allFilesName:
		print f

def seperateDiffPath(allFilesName):
	global centreNodes
	sepWithCentre = []
	entityCentre = []
	for f in allFilesName:
		lf = f.split('.')
		num = int(lf[0])
		if num in centreNodes:
			entityCentre.append(f)
		else:
			if entityCentre:
				sepWithCentre.append(entityCentre)
				entityCentre = []
				centreNodes.append(num)
				entityCentre.append(f)
			else:
				centreNodes.append(num)
				entityCentre.append(f)
	return sepWithCentre

def showSep(object):
	for item in object:
		if isinstance(item, list):
			for i in item:
				print i
			print ''
		else:
			print item

def singleProcess(object):
	# 对同一个nodeId.postfix的文件
	# 读取circles
	# 先读取featnames, 可以获取features的个数（列数）
	# 然后读取feat
	global adj_node
	cur_dict = {}
	for i in object:
		lf = i.split('.')
		if lf[1] == 'featnames':
			with open(path + '\\' + i, 'r') as f:
				for line in f.readlines():
					lf1 = line.split(' ')
					cur_dict[int(lf1[0])] = lf1[1]

	if cur_dict:
		# 如果cur_dict不为空
		featname.append(cur_dict)
	else:
		raise Exception('no featname file')

	# 获取特征维度
	count = len(cur_dict)
	# 每一行表示一个向量
	cur_feat = np.zeros((4039, count))
	
	# 然后读取feat文件
	for i in object:
		lf = i.split('.')
		if lf[1] == 'egofeat':
			with open(path + '\\' + i, 'r') as f:
				for line in f.readlines():
					llf = line.split(' ')
					cur_id = int(llf[0])
					for t in range(1, len(llf)):
						cur_feat[cur_id][t-1] = llf[t]

		elif lf[1] == 'feat':
			with open(path + '\\' + i, 'r') as f:
				for line in f.readlines():
					llf = line.split(' ')
					cur_id = int(llf[0])
					for t in range(1, len(llf)):
						cur_feat[cur_id][t-1] = llf[t]

	# 然后读取edges文件 读取到一行链接边的时候 用两个features向量计算距离
	# 用欧几里得距离公式
	cur_adj = np.zeros((4039,4039))
	for i in object:
		lf = i.split('.')
		if lf[1] == 'edges':
			with open(path + '\\' + i, 'r') as f:
				for line in f.readlines():
					llf = line.split(' ')
					id1 = int(llf[0])
					id2 = int(llf[1])
					cur_adj[id1][id2] = np.sqrt(np.sum(np.square(cur_feat[id1] - cur_feat[id2])))
					# print cur_adj[id1][id2]
				
				adj_node = 0.5 * adj_node + 0.5 * cur_adj
	
# 实现spectral clustering
# 以下是kmeans相关算法


def kmeans(data, k=2):
    def _distance(p1, p2):
        """
        计算两个向量之间的欧几里得距离
        """
        dis = np.sum((p1-p2)**2)
        return np.sqrt(dis)


    def _rand_center(data, k):
        """随机产生K个质心"""
        n = data.shape[1] # features
        centroids = np.zeros((k,n)) # init with (0,0)....
        for i in range(n):
            dmin, dmax = np.min(data[:,i]), np.max(data[:,i]) # 第i列的最大最小值
            centroids[:,i] = dmin + (dmax - dmin) * np.random.rand(k)
        return centroids
    
    def _converged(centroids1, centroids2):
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return (set1 == set2)
    
    n = data.shape[0]
    centroids = _rand_center(data, k)
    label = np.zeros(n ,dtype = np.int)
    assement = np.zeros(n)
    converged = False
    count = 0
    
    while not converged and count < 200:
        print count
        count += 1
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist = np.inf
            for j in range(k):
                dist = _distance(data[i], centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    label[i] = j
            assement[i] = _distance(data[i], centroids[label[i]])**2
                    
        # update centroid
        for m in range(k):
            centroids[m] = np.mean(data[label == m], axis = 0)
        converged = _converged(old_centroids, centroids)
    return centroids, label, np.sum(assement)
    
if __name__ == '__main__':
    allFilesName = getFilesName(path)
    # printFilesName(allFilesName)
    sepWithCentre = seperateDiffPath(allFilesName)
    count = len(centreNodes)
    # showSep(sepWithCentre)

    """
    .edges 文件: 从node.id看出去的所有结点和对应的边连接关系
    .circles 文件: 从node.id视角看出去 确定存在的环
    .feats 文件:  其它结点对应的features
    .featname 文件:  每个feature对应的名字
    .egofeat 文件:  当前结点对应的features

    文件读取顺序: 对每个中心节点（包含一个从这个节点视角看出去的一个网络图），先读取这个网络图分析的features（特征用于构建距离）
    不同的中心节点的视角网络所含的特征数目不同。然后读取edges文件，每次遇到相邻的两个节点则计算这两个结点之间的距离。填入矩阵。
    featname并没有什么卵用，只是用来最后分析出占主要影响的因素是哪一个feature。
    circles文件则用于分析聚类效果。
    """
	# 打开不同的文件 读取结点数目和边构造网络图
    for centrenodesfile in sepWithCentre:
        singleProcess(centrenodesfile)

	# 根据motif构建邻接矩阵 然后进行spectral clustering
	# adj_node 表示 节点之间的连接关系
    # spectral clustering
    # 把adj_node转成对称矩阵
    adj_node = np.triu(adj_node)
    adj_node += adj_node.T - np.diag(adj_node.diagonal())
    W = adj_node
    D = np.zeros((len(W), len(W)))
    for i in range(0, len(W)):
        D[i][i] = np.sum(W[i])
    
    # 构建拉普拉斯矩阵
    L = D - W
    # 计算拉普拉斯矩阵的特征值和特征向量
    a, b = np.linalg.eig(L)
    # 排序
    c = list(zip(a, b))
    d = sorted(c, key = lambda x : x[0])
    e, f = list(zip(*d))
    # 去掉那些接近于0的特征值
    ix = 0
    for i in range(len(e)):
        if np.sum(e[:i+1])>np.sum(e)*0.01:
            ix = i
            break
    use_a = e[ix:]
    use_b = f[ix:]

    # 把特征向量都转成矩阵
    ls = use_b[0].reshape(4039, 1)
    for i in range(1, len(use_b)):
        temp = use_b[i].reshape(4039, 1)
        ls = np.concatenate((ls, temp), axis = 1)
    # 对b的每一行进行kmeans聚类
    print ls
    
    res_centroids, res_label, res_ass = kmeans(ls, 5)