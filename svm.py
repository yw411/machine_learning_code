import numpy as np

class SVM(object):

    def __init__(self,C,kernel,tor=10e-4,maxiter=200):
        self.C=C
        self.k=kernel   #kernel is tuple
        self.tor=tor
        self.maxiter=maxiter

    def cal_g(self,i):
        return np.sum(self.alpha*self.Y*self.k_gram[i,:])+self.b

    def kkt(self,i):
        '''
        a = 0 = > outside
        0 < a < c = > on
        the
        margin, active
        support
        a = c = > inside
        $y_i
        E_i = y_i(g_i - y_i) = y_ig_i - 1$
        '''
        ei=self.Y[i]*self.cal_g(i)-1
        return (np.isclose(self.alpha[i],0.0,atol=self.tor) and ei>=0) or  \
                (0<self.alpha[i]<self.C and np.isclose(ei,0.0,atol=self.tor)) or \
               (np.isclose(self.alpha[i],self.C,atol=self.tor) and ei<=0)

    def kernel_cal(self,data1,data2):
        m,n=data1.shape
        name=self.k[0]
        result=np.zeros(m,dtype=np.float32)
        if name=='linear':
            return np.matmul(data1,data2.T)
        elif name=='rbf':#for d in data2
            for i in range(m):
                delta=data2-data1[i,:]
                result[i]=np.matmul(delta,delta.T)
            result=np.exp(result/(-2*self.k[1]*self.k[1]))
            return result






    def selectij(self,m):
        active = []
        other = []
        for i in range(m):  #select zhichixiangliang first
            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                active.append(i)
            else:
                other.append(i)

        for i in active + other:
            if not self.kkt(i):  #if i is all kkt,then can stop smo
                ei = self.cal_g(i) - self.Y[i]
                # select a j
                max_delta = 0
                max_j = -1
                max_new_j=0
                max_ej=0
                for idx, j in enumerate(active + other):
                    ej = self.cal_g(j) - self.Y[j]
                    aiold = self.alpha[i].copy()
                    ajold = self.alpha[j].copy()
                    yita = self.k_gram[i][i] + self.k_gram[j][j] - 2 * self.k_gram[i][j]
                    if yita <= 0:
                        continue
                    if self.Y[i] == self.Y[j]:
                        L = max(0, aiold + ajold - self.C)
                        H = min(self.C, aiold + ajold)
                    else:
                        L = max(0, ajold - aiold)
                        H = min(self.C, self.C + ajold - aiold)
                    if L == H:
                        continue
                    ajnew = ajold + self.Y[j] * (ei - ej) / yita
                    if ajnew > H:
                        ajnew = H
                    elif ajnew < L:
                        ajnew = L
                    delta = abs(ajnew - ajold)
                    if delta > max_delta:
                        max_delta = delta
                        max_j = j
                        max_new_j=ajnew
                        max_ej=ej
                    if idx + 1 == len(active) and not np.isclose(max_delta, self.tor):   #else  find it in not zhichixiangliang
                        break
                if np.isclose(max_delta, self.tor):  #no satisfy, find another i
                    continue
                else:
                    #update i
                    ainew=self.alpha[i]+self.Y[i]*self.Y[max_j]*(self.alpha[max_j]-max_new_j)

                    #updata b
                    b1=self.b-ei-self.Y[i]*self.k_gram[i][i]*(ainew-self.alpha[i])-self.Y[max_j]*self.k_gram[max_j][i]*(max_new_j-self.alpha[max_j])
                    b2=self.b-max_ej-self.Y[i]*self.k_gram[i][max_j]*(ainew-self.alpha[i])-self.Y[max_j]*self.k_gram[max_j][max_j]*(max_new_j-self.alpha[max_j])

                    if ainew>0 and ainew<self.C:
                        self.b=b1
                    elif max_new_j>0 and max_new_j<self.C:
                        self.b=b2
                    else:
                        self.b=(b1+b2)/2
                    self.alpha[i]=ainew
                    self.alpha[max_j]=max_new_j
                    return max_j
        return None


    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        m,n=self.X.shape
        self.alpha=np.zeros([m])
        self.b=0

        #calculate k
        self.k_gram=np.zeros((m,m),dtype=np.float32)
        for i in range(m):
            self.k_gram[i,:]=self.kernel_cal(self.X,self.X[i,:])
        #smo
        for iter in range(self.maxiter):
            rep=self.selectij(m)
            if rep==None:
                break

    def predict(self,test):
        n,m=test.shape
        pre=np.zeros(n,dtype=np.int64)
        for i in range(n):
            cur_x=test[i]
            cur_pre=np.sign(np.sum(self.alpha*self.Y*self.kernel_cal(self.X,cur_x))+self.b)
            pre[i]=cur_pre
        return pre






