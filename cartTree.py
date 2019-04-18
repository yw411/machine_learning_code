import numpy as np

class Node(object):
    def __init__(self,fea,val,left,right,quzhi):
        self.feature=fea
        self.value=val
        self.left=left
        self.right=right
        self.quzhi=quzhi

def mse_error(y):
        y_mean=np.mean(y)
        return np.var(y)*y.shape[0]

class CartTreeRegression(object):

    def __init__(self,ops=(1,4),errCal=mse_error):
        self.ops=ops
        self.errCal=errCal
        return

    def fit(self,X,Y):
        self.tree=self.creatTree(X,Y)

    def splitData(self,x,y,fea,value):
        n=len(y)
        index_right=np.nonzero(x[:,fea]>value)[0]
        index_left=np.nonzero(x[:,fea]<=value)[0]
        if len(index_right)==0:
            return x[index_left,:],y[index_left],np.array([[]]),np.array([])
        elif len(index_left)==0:
            return np.array([[]]),np.array([]),x[index_right,:],y[index_right]
        else:
            return x[index_left,:],y[index_left],x[index_right,:],y[index_right]


    def mse_error(self,y):
        y_mean = np.mean(y)
        return np.var(y) * y.shape[0]

    def findBest(self,x,y,errCal,ops):
        tolS=ops[0]
        tolN=ops[1]  #min_leaf
        n,m=x.shape
        bestValue=0
        bestFeature=0
        bestError=np.inf

        #print (y)
        oldErr=self.mse_error(y)
        if len(set(y))==1:  #bukefen
            return None,np.mean(y)
        for f in range(m):
            values=set(x[:,f])
            for v in values:
                left_x, left_y, right_x, right_y=self.splitData(x,y,f,v)
                if (len(left_y)<tolN) or (len(right_y)<tolN):  #bukefen,continue to find
                    continue
                total_error=self.mse_error(left_y)+self.mse_error(right_y)
                if total_error<bestError:
                    bestError=total_error
                    bestFeature=f
                    bestValue=v
        if (oldErr-bestError)<tolS:  #after split, the error increase  no split
            return None,np.mean(y)

        left_x, left_y, right_x, right_y = self.splitData(x, y, bestFeature,bestValue)
        if (len(left_y) < tolN) or (len(right_y) < tolN):  # if continue always,
            return None,np.mean(y)
        return bestFeature,bestValue


    def creatTree(self,x,y):  #bukefen
        #find best feature and value
        feature,value=self.findBest(x,y,self.errCal,self.ops)
        if feature is None:
            return Node(None,None,None,None,value)
        left_x,left_y,right_x,right_y=self.splitData(x,y,feature,value)
        rightnode=self.creatTree(right_x,right_y)
        leftnode=self.creatTree(left_x,left_y)
        node=Node(feature,value,leftnode,rightnode,-1)
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

if __name__=='__main__':
    f=open('machinelearninginaction/Ch09/bikeSpeedVsIq_train.txt')
    data=f.readlines()
    f.close()

    x=[]
    y=[]
    for d in data:
        cur_x=[]
        line=d.strip().split('\t')
        y.append(float(line[1]))
        cur_x.append(float(line[0]))
        #cur_x.append(float(line[1]))
        x.append(cur_x)
    xa=np.array(x)
    ya=np.array(y)

    model=CartTreeRegression()
    model.fit(xa,ya)

    f = open('machinelearninginaction/Ch09/bikeSpeedVsIq_test.txt')
    data = f.readlines()
    f.close()

    x = []
    y = []
    for d in data:
        cur_x = []
        line = d.strip().split('\t')
        y.append(float(line[1]))
        cur_x.append(float(line[0]))
        # cur_x.append(float(line[1]))
        x.append(cur_x)
    xat = np.array(x)
    yat = np.array(y)

    test=[]
    for x in xat:
        test.append(model.predict(x))
    testa=np.array(test)
    r=np.corrcoef(yat,testa,rowvar=0)[0,1]
    print (r)





