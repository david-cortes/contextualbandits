import numpy as np, types
from copy import deepcopy


def _modify_predict_method(classifier):
    classifier.predict_old=deepcopy(classifier.predict)
    classifier.predict=types.MethodType(_robust_predict, classifier)

    if 'predict_proba' in dir(classifier):
        # TODO: once scikit-learn's issue 10938 is solved, make like the others
        classifier.predict_proba_new=types.MethodType(_robust_predict_proba, classifier)
    if 'decision_function' in dir(classifier):
        classifier.decision_function_old=classifier.decision_function
        classifier.decision_function=types.MethodType(_robust_decision_function, classifier)

    return classifier

def _robust_predict(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.predict_old(X)
        except:
            return np.zeros(X.shape[0])
    else:
        return np.zeros(X.shape[0])

def _robust_predict_proba(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.predict_proba(X)
        except:
            return np.zeros((X.shape[0],2))
    else:
        return np.zeros((X.shape[0],2))

def _robust_decision_function(self, X):
    if ('coef_' in dir(self)) or ('coefs_' in dir(self)):
        try:
            return self.decision_function_old(X)
        except:
            return np.zeros(X.shape[0])
    else:
        return np.zeros(X.shape[0])

def _check_bools(batch_train=False,smooth_predictions=False,assume_unique_reward=False):
    assert isinstance(batch_train, bool)
    assert isinstance(smooth_predictions, bool)
    assert isinstance(assume_unique_reward, bool)
    return batch_train,smooth_predictions,assume_unique_reward

def _check_constructor_input(base_algorithm,nchoices,batch_train=False):
    assert nchoices>2
    assert isinstance(nchoices, int)
    assert ('fit' in dir(base_algorithm)) and ('predict' in dir(base_algorithm))
    if batch_train:
        assert 'partial_fit' in dir(base_algorithm)
    return None

def _check_beta_prior(beta_prior, nchoices, default_b):
    if beta_prior=='auto':
        out = (_calculate_beta_prior(nchoices),default_b)
    elif beta_prior==None:
        out=((1,1),0)
    else:
        assert len(beta_prior)==2
        assert len(beta_prior[0])==2
        assert isinstance(beta_prior[1],int)
        assert isinstance(beta_prior[0][0],int) or isinstance(beta_prior[0][0],float)
        assert isinstance(beta_prior[0][1],int) or isinstance(beta_prior[0][1],float)
        assert (beta_prior[0][0]>0) and (beta_prior[0][1]>0)
        out = beta_prior
    return out

def _check_fit_input(X,a,r):
    X=_check_X_input(X)
    a=_check_1d_inp(a)
    r=_check_1d_inp(r)
    assert X.shape[0]==a.shape[0]
    assert X.shape[0]==r.shape[0]
        
    return X,a,r

def _check_X_input(X):
    if type(X).__name__=='DataFrame':
        X=X.values
    if type(X)==np.matrixlib.defmatrix.matrix:
        X=np.array(X)
    if type(X)!=np.ndarray:
        raise ValueError("'X' must be a numpy array or pandas data frame.")
    if len(X.shape)==1:
        X=X.reshape(1,-1)
    assert len(X.shape)==2
    return X

def _check_1d_inp(y):
    if type(y).__name__=='DataFrame' or type(y).__name__=='Series':
        y=y.values
    if type(y)==np.matrixlib.defmatrix.matrix:
        y=np.array(y)
    if type(y)!=np.ndarray:
        raise ValueError("'a' and 'r' must be numpy arrays or pandas data frames.")
    if len(y.shape)==2:
        assert y.shape[1]==1
        y=y.reshape(-1)
    assert len(y.shape)==1
    return y

class _BetaPredictor:
    def __init__(self,a,b):
        self.a=a
        self.b=b
        
    def fit(self,X=None,y=None):
        pass
    
    def predict_proba(self,X):
        preds = np.random.beta(self.a, self.b, size=X.shape[0]).reshape(-1,1)
        return np.c_[1-preds,preds]
    
    def decision_function(self,X):
        return np.random.beta(self.a, self.b, size=X.shape[0])
    
    def predict(self,X):
        return (np.random.beta(self.a, self.b, size=X.shape[0])).astype('uint8')
    
    def predict_avg(self,X):
        pred=self.decision_function(X)
        return -np.log((1-pred)/pred)
    
    def predict_rnd(self,X):
        return self.predict_avg(X)
    
    def predict_ucb(self,X):
        return self.predict_avg(X)
    
class _ZeroPredictor:
    def __init__(self):
        pass
        
    def fit(self,X=None,y=None):
        pass
    
    def predict_proba(self,X):
        return np.c_[np.ones((X.shape[0],1)),np.zeros((X.shape[0],1))]
    
    def decision_function(self,X):
        return np.zeros(X.shape[0])
    
    def predict(self,X):
        return np.zeros(X.shape[0])
    
    def predict_avg(self,X):
        return np.zeros(X.shape[0])
    
    def predict_rnd(self,X):
        return np.zeros(X.shape[0])
    
    def predict_ucb(self,X):
        return np.zeros(X.shape[0])
    
class _OnePredictor:
    def __init__(self):
        pass
        
    def fit(self,X=None,y=None):
        pass
    
    def predict_proba(self,X):
        return np.c_[np.zeros((X.shape[0],1)),np.ones((X.shape[0],1))]
    
    def decision_function(self,X):
        return np.ones(X.shape[0])
    
    def predict(self,X):
        return np.ones(X.shape[0])
    
    def predict_avg(self,X):
        return np.ones(X.shape[0])
    
    def predict_rnd(self,X):
        return np.ones(X.shape[0])
    
    def predict_ucb(self,X):
        return np.ones(X.shape[0])
    
class _RandomPredictor:
    def __init__(self):
        pass
    
    def fit(self,X=None, y=None, sample_weight=None):
        pass
    
    def predict(self,X):
        return (np.random.random(size=X.shape[0])>=.5).astype('uint8')
    
    def predict_proba(self,X):
        return np.random.random(size=X.shape[0])
    
    def decision_function(self,X):
        pred=self.predict_proba(X)
        return -np.log((1-pred)/pred)

class _ArrBSClassif:
    def __init__(self,base,X,a,r,n,thr,alpha,beta,samples,smooth=False,assume_un=False,
                 partialfit=False,partial_method='gamma'):
        if partialfit:
            base=_modify_predict_method(base)
        self.base=base
        self.algos=[[deepcopy(base) for i in range(samples)] for j in range(n)]
        self.n=n
        self.samples=samples
        self.partialfit=partialfit
        self.partial_method=partial_method
        self.smooth=smooth
        self.assume_un=assume_un
        if self.smooth:
            self.counters=np.zeros((1,n))

        if self.partialfit:
            self.partial_fit(X,a,r)
        else:
            for choice in range(n):
                if self.assume_un:
                    this_choice=(a==choice)
                    arms_w_rew=(r==1)
                    yclass=r[this_choice|arms_w_rew]
                    yclass[arms_w_rew&(~this_choice)]=0
                    this_choice=this_choice|arms_w_rew
                else:
                    this_choice=(a==choice)
                    yclass=r[this_choice]
                if self.smooth:
                    self.counters[0,choice]+=yclass.shape[0]

                n_pos=yclass.sum()
                if n_pos<thr:
                    self.algos[choice]=_BetaPredictor(alpha+n_pos,beta+yclass.shape[0]-n_pos)
                    continue
                if (n_pos==0) and (thr<1):
                    self.algos[choice]=_ZeroPredictor()
                    continue
                xclass=X[this_choice,:]
                for sample in range(samples):
                    ix_take=np.random.randint(xclass.shape[0], size=xclass.shape[0])
                    xsample=xclass[ix_take,:]
                    ysample=yclass[ix_take]
                    nclass=ysample.sum()
                    if nclass==ysample.shape[0]:
                        self.algos[choice][sample]=_OnePredictor()
                    elif ysample.sum()>0:
                        self.algos[choice][sample].fit(xsample,ysample)
                    else:
                        self.algos[choice][sample]=_ZeroPredictor()
                    
    def partial_fit(self,X,a,r):
        for choice in range(self.n):
            if self.assume_un:
                this_choice=(a==choice)
                arms_w_rew=(r==1)
                yclass=r[this_choice|arms_w_rew]
                yclass[arms_w_rew&(~this_choice)]=0
                this_choice=this_choice|arms_w_rew
            else:
                this_choice=(a==choice)
                yclass=r[this_choice]

            if self.smooth:
                self.counters[0,choice]+=yclass.shape[0]

            xclass=X[this_choice,:]
            
            if xclass.shape[0]>0:
                for sample in range(self.samples):
                    if self.partial_method=='poisson':
                        appear_times=np.random.poisson(1, size=xclass.shape[0])
                        xsample=np.repeat(xclass, appear_times)
                        ysample=np.repeat(yclass, appear_times)
                        self.algos[choice][sample].partial_fit(xsample,ysample,classes=[0,1])
                    else:
                        self.algos[choice][sample].partial_fit(xclass,yclass,
                                classes=[0,1],sample_weight=np.random.gamma(2,2,size=xclass.shape[0]))
    
    def score_avg(self,X):
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            if not (type(self.algos[choice])==list):
                preds[:,choice]=self.algos[choice].decision_function(X)
                continue
            for sample in range(self.samples):
                if 'predict_proba_new' in dir(self.algos[choice][sample]):
                    preds[:,choice]+=self.algos[choice][sample].predict_proba_new(X)[:,1]
                elif 'predict_proba' in dir(self.algos[choice][sample]):
                    preds[:,choice]+=self.algos[choice][sample].predict_proba(X)[:,1]
                elif 'decision_function' in dir(self.algos[choice][sample]):
                    preds[:,choice]+=self.algos[choice][sample].decision_function(X)
                else:
                    preds[:,choice]+=self.algos[choice][sample].predict(X)
                
            preds[:,choice]=preds[:,choice]/self.samples
        
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
        
        return preds
    
    def score_max(self,X,perc=100):
        global preds, arr_compare, choice, qperc
        qperc=perc
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            if not (type(self.algos[choice])==list):
                preds[:,choice]=self.algos[choice].decision_function(X)
                continue
            arr_compare=list()
            for sample in range(self.samples):
                if 'predict_proba_new' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].predict_proba_new(X)[:,1])
                elif 'predict_proba' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].predict_proba(X)[:,1])
                elif 'decision_function' in dir(self.algos[choice][sample]):
                    arr_compare.append(self.algos[choice][sample].decision_function(X))
                else:
                    raise Exception("BootstrappedUCB requires a classifier with method 'predict_proba' or 'decision_function'")
                
            if perc==100:
                preds[:,choice]=np.vstack(arr_compare).max(axis=0)
            else:
                preds[:,choice]=np.percentile(np.vstack(arr_compare), perc, axis=0)
        
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
        
        return preds
    
    def score_rnd(self,X):
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            if not (type(self.algos[choice])==list):
                preds[:,choice]=self.algos[choice].decision_function(X)
                continue
            sample_take=np.random.randint(self.samples)
            preds[:,choice]=self.algos[choice][sample_take].predict(X)
        
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
        
        return preds
    
class _OneVsRest:
    def __init__(self,base,X,a,r,n,thr,alpha,beta,smooth=False,assume_un=False,partialfit=False):
        self.n=n
        if partialfit:
            base=_modify_predict_method(base)
        self.base=base
        self.algos=[deepcopy(base) for i in range(n)]
        self.smooth=smooth
        self.assume_un=assume_un
        if self.smooth:
            self.counters=np.zeros((1,n))
            
        if partialfit:
            self.partial_fit(X,a,r)
        else:
            for choice in range(n):
                if self.assume_un:
                    this_choice=(a==choice)
                    arms_w_rew=(r==1)
                    yclass=r[this_choice|arms_w_rew]
                    yclass[arms_w_rew&(~this_choice)]=0
                    this_choice=this_choice|arms_w_rew
                else:
                    this_choice=(a==choice)
                    yclass=r[this_choice]
                n_pos=yclass.sum()
                if self.smooth:
                    self.counters[0,choice]+=yclass.shape[0]
                if n_pos<thr:
                    self.algos[choice]=_BetaPredictor(alpha+n_pos,beta+yclass.shape[0]-n_pos)
                    continue
                if n_pos==yclass.shape[0]:
                    self.algos[choice]=_OnePredictor()
                    continue
                if (n_pos==0) and (thr<1):
                    self.algos[choice]=_ZeroPredictor()
                xclass=X[this_choice,:]
                self.algos[choice].fit(xclass,yclass)
                
    def partial_fit(self,X,a,r):
        for choice in range(self.n):
            if self.assume_un:
                this_choice=(a==choice)
                arms_w_rew=(r==1)
                yclass=r[this_choice|arms_w_rew]
                yclass[arms_w_rew&(~this_choice)]=0
                this_choice=this_choice|arms_w_rew
            else:
                this_choice=(a==choice)
                yclass=r[this_choice]

            if self.smooth:
                self.counters[0,choice]+=yclass.shape[0]

            xclass=X[this_choice,:]
            if xclass.shape[0]>0:
                self.algos[choice].partial_fit(xclass,yclass,classes=[0,1])
            
            
    
    def decision_function(self,X):
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            try:
                preds[:,choice]=self.algos[choice].decision_function(X)
            except:
                try:
                    preds[:,choice]=self.algos[choice].predict_proba(X)[:,1]
                except:
                    preds[:,choice]=self.algos[choice].predict_proba_new(X)[:,1]
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
        return preds
    
    def predict_proba(self,X):
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            if 'predict_proba_new' in dir(self.base):
                preds[:,choice]=self.algos[choice].predict_proba_new(X)[:,1]
            elif 'predict_proba' in dir(self.base):
                preds[:,choice]=self.algos[choice].predict_proba(X)[:,1]
            elif 'decision_function' in dir(self.base):
                preds[:,choice]=self.algos[choice].decision_function(X)
            else:
                raise Exception("Softmax exploration requires a classifier with 'predict_proba' or 'decision_function' method.")
        
        if 'predict_proba' in dir(self.base):
            preds=preds/preds.sum(axis=1).reshape(-1,1)
        else:
            preds=np.exp(preds - preds.max(axis=1).reshape(-1,1))
            preds=preds/preds.sum(axis=1).reshape(-1,1)
            
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
            
        return preds
    
    def predict_proba_raw(self,X):
        preds=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            if 'predict_proba_new' in dir(self.algos[choice]):
                preds[:,choice]=self.algos[choice].predict_proba_new(X)[:,1]
            else:
                preds[:,choice]=self.algos[choice].predict_proba(X)[:,1]
                
        if self.smooth:
            preds=(preds*self.counters+1)/(self.counters+2)
                
        return preds
    
    def predict(self,X):
        return np.argmax(self.decision_function(X), axis=1)
    
class _BayesianLogisticRegression:
    def __init__(self,X,y,method='advi'):
        import pymc3 as pm, pandas as pd
        X=_check_X_input(X)
        y=_check_1d_inp(y)
        assert X.shape[0]==y.shape[0]

        with pm.Model():
            pm.glm.linear.GLM(X,y,family='binomial')
            pm.find_MAP()
            if method=='advi':
                trace=pm.fit(progressbar=False)
            if method=='nuts':
                trace=pm.sample(progressbar=False)
        if method=='advi':
            coefs=[i for i in trace.sample()]
        if method=='nuts':
            coefs=[i for i in trace]
        coefs=pd.DataFrame.from_dict(coefs)
        coefs=coefs[['Intercept']+['x'+str(i) for i in range(X.shape[1])]]
        coefs=coefs.values
        coefs=coefs.T
        
    def predict_all(self,X):
        X_w_bias=np.c_[np.ones((X.shape[0],1)), X]
        return X_w_bias.dot(coefs)
    
    def predict_avg(self,X):
        pred=self.predict_all(X)
        return pred.mean(axis=1)
    
    def predict_ucb(self,X,p):
        pred=self.predict_all(X)
        return np.percentile(pred,p, axis=1).reshape(-1)
    
    def predict_rnd(self,X):
        pred=self.predict_all(X)
        col_take=np.random.randint(pred.shape[1], size=X.shape[0])
        pred=pred[np.arange(X.shape[0]),col_take]
        return pred.reshape(-1)
    
class _BayesianOneVsRest:
    def __init__(self,X,a,r,n,thr,alpha,beta,method):
        self.n=n
        self.algos=[None for i in range(n)]
        for choice in range(n):
            this_choice=(a==choice)
            yclass=r[this_choice]
            n_pos=yclass.sum()
            if n_pos<thr:
                self.algos[choice]=_BetaPredictor(alpha+n_pos,beta+yclass.shape[0]-n_pos)
                continue
            xclass=X[this_choice,:]
            self.algos[choice]=_BayesianLogisticRegression(xclass,yclass,method)
            
    def predict_ucb(self,X,p):
        pred=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            pred[:,choice]=self.algos[choice].predict_ucb(X,p)
        return pred

    def predict_rnd(self,X):
        pred=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            pred[:,choice]=self.algos[choice].predict_rnd(X)
        return pred

    def predict_avg(self,X):
        pred=np.zeros((X.shape[0],self.n))
        for choice in range(self.n):
            pred[:,choice]=self.algos[choice].predict_avg(X)
        return pred

def _calculate_beta_prior(nchoices):
    return (3/nchoices,4)

class _LinUCBSingle:
    def __init__(self, alpha):
        self.alpha=alpha
        
    def fit(self, X, y):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        self.Ainv=np.eye(X.shape[1])
        self.b=np.zeros((X.shape[1], 1))
        
        self.partial_fit(X,y)
    
    def partial_fit(self, X, y):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        if 'Ainv' not in dir(self):
            self.Ainv=np.eye(X.shape[1])
            self.b=np.zeros((X.shape[1], 1))
        sumb=np.zeros((X.shape[1], 1))
        for i in range(X.shape[0]):
            x=X[i,:].reshape(-1,1)
            r=y[i]
            sumb+=r*x
            
            self.Ainv -= np.linalg.multi_dot([self.Ainv, x, x.T, self.Ainv])/\
                            (1 + np.linalg.multi_dot([x.T, self.Ainv, x]))
        
        self.b+=sumb
    
    def predict(self, X, exploit=False):
        if len(X.shape)==1:
            X=X.reshape(1,-1)
        
        pred=self.Ainv.dot(self.b).T.dot(X.T).reshape(-1)
        
        if not exploit:
            return pred
        
        for i in range(X.shape[0]):
            x=X[i,:].reshape(-1,1)
            cb=self.alpha*np.sqrt(np.linalg.multi_dot([x.T, self.Ainv, x]))
            pred[i]+=cb[0]
        
        return pred
