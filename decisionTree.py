import  numpy as np

class dt(object):

    def __init__(self,method='id3',eps=10e-4):
        self.tree=dict()
        self.method=method
        self.eps=eps

    def fit(self,x,y):
        self.tree=self.buildTree(x,y)

    def cal_entropy(self,y):
        entropy=0
        total=len(y)
        y_list=list(y)
        y_value=set(y)
        for v in y_value:
            curnum=y_list.count(v)
            p=curnum*1.0/total
            entropy=entropy-p*np.log2(p)
        return entropy




    def buildTree(self,x,y):   #need digui   x,y are both array
        tree=dict()
        n,m=x.shape
        if len(set(y))==1:
            return y[0]
        elif m==0:
            return max(y)
        oldEntroy=self.cal_entropy(y)
        bestGain=0
        bestFeature=0
        for f in range(m):
            cur_entropy=self.split_data_entropy(x,y,f)
            gain=oldEntroy-cur_entropy
            if gain>bestGain:
                bestGain=gain
                bestFeature=f
        if bestGain<self.eps:
            return max(y)
        else:  #digui
            tree['feature']=bestFeature
            values=set(x[:,bestFeature])
            tree['values']=list(values)
            tree['children']=[]
            for v in values:
                cur_index=np.nonzero(x[:,bestFeature]==v)[0]
                cur_x=x[cur_index]
                cur_y=y[cur_y]
                cur_x=np.delete(cur_x,bestFeature,1)
                tree['children'].append(self.buildTree(cur_x,cur_y))
        return tree

    def split_data_entropy(self,x,y,f):
        total_entropy=0
        values = set(x[:, f])
        for v in values:
            cur_index=np.nonzero(x[:, f] ==v)[0]
            cury=y[cur_index]
            total_entropy=total_entropy+(len(cury)*1.0/len(y))*self.cal_entropy(cury)
        return total_entropy

    def predict(self,x):
        pass