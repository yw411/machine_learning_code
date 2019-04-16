import numpy as np

class Node(object):
    def __init__(self,fea,val,left,right,quzhi):
        self.feature=fea
        self.value=val
        self.left=left
        self.right=right
        self.quzhi=quzhi

class CartTreeRegression(object):

    def __init__(self,ops,errCal):
        self.ops=ops
        self.errCal=errCal
        return

    def fit(self,X,Y):
        self.tree=self.creatTree(X,Y)

    def splitData(self,x,y,fea,value):
        index_right=np.nonzero(x[:,fea]>value)[0]
        index_left=np.nonzero(x[:,fea]<=value)[0]
        return x[index_left,:],y[index_left],x[index_right,:],y[index_right]

    def mse_error(self,y):
        y_mean=np.mean(y)
        return np.var(y)*y.shape[0]

    def findBest(self,x,y,errCal=mse_error,ops=(1,4)):
        tolS=ops[0]
        tolN=ops[1]  #min_leaf
        n,m=x.shape
        bestValue=0
        bestFeature=0
        bestError=np.inf

        oldErr=errCal(y)
        if len(set(y))==1:  #bukefen
            return None,np.mean(y)
        for f in range(m):
            values=set(x[:,f])
            for v in values:
                left_x, left_y, right_x, right_y=self.splitData(x,y,f,v)
                if (len(left_y)<tolN) or (len(right_y)<tolN):  #bukefen,continue to find
                    continue
                total_error=errCal(left_y)+errCal(right_y)
                if total_error<bestError:
                    bestError=total_error
                    bestFeature=f
                    bestValue=v
        if (oldErr-bestError)<tolS:  #after split, the error increase  no split
            return None,np.mean(y)

        left_x, left_y, right_x, right_y = self.splitData(x, y, bestFeature,bestValue)
        if (len(left_y) < tolN) or (len(right_y) < tolN):  # if continue always,
            return None,np.mean(y)
        return f,v


    def creatTree(self,x,y):  #bukefen
        #find best feature and value
        feature,value=self.findBest(x,y,self.errCal,self.ops)
        if feature is None:
            return Node(None,None,None,None,value)
        left_x,left_y,right_x,right_y=self.splitData(x,y,feature,value)
        leftnode=self.creatTree(left_x,left_y)
        rightnode=self.creatTree(right_x,right_y)
        node=Node(feature,value,leftnode,rightnode)
        return node

    def predict(self,test):  #for a instance
        node=self.tree
        while(node.left is not None):
            f=node.feature
            cur_value=test[f]
            if cur_value<=node.value:
                node=node.left
            else:
                node=node.right

        return node.quzhi
