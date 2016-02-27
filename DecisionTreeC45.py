import pandas as pd
import numpy as np
import collections

def I(counts):
        i = 0
        T = sum(counts)
        for c in counts:
            i += -(c/T)*np.log2(c/T)
        return(i)
    


class DecisionTree(object):
    
    #Instances of DecisionTree objects must initialize with a pointer to a pandas dataframe
    def __init__(self, df):
        self.df = df
    
    #main method for DecisionTree fitting, with optional parameters
    def fit(self, yname, Xnames, Xnum = None, loss = "C4.5", minleaves = 1, maxdepth = None):
        
        self.loss = loss
        self.minleaves = minleaves
        self.maxdepth = maxdepth
        
        if yname in self.df.keys():
            self.y = yname
        else:
            raise NameError('%s not in Dataframe' % yname)
        self.x = []
        
        for xn in Xnames:
            if xn in self.df.keys():
                self.x.append(xn)
            else:
                raise NameError('%s not in Dataframe' % xn)
        
        self.ygroup = self.df.groupby(self.y).groups
        self.ycc = {}
        for key in self.ygroup:
            self.ycc[key] = len(self.ygroup[key])
        self.nid = 0
        self.nodes = {}
        self.queue = collections.deque()
        self.queue.append(Node([], self.x, self.ycc))
        
        while self.queue:
            
            node_i = self.queue.popleft()
            
            Info_y = I(node_i.yc_i.values())
            
            ct = self.ContinGENcyTable(node_i.pathkeys, node_i.Xnames)
            maxgain = 0
            
            T = sum(node_i.yc_i.values())
            for c in ct:
                Info_yx = np.sum(np.multiply(
                        np.apply_along_axis(I, 1, c[1]),
                        np.apply_along_axis(np.sum, 1, c[1]) / T
                        ))
                Gain_x = Info_y - Info_yx
                if Gain_x > maxgain:
                    self.chosen = c
                    maxgain = Gain_x
            print((self.chosen, maxgain))
            node_i.opt = c[0]
            node_i.levels = c[2]
            for i in range(len(node_i.levels)):
                path = node_i.pathkeys[:] + [(c[0],'=',c[2][i])]
                ycci = {}
                for j, key in enumerate(self.ygroup):
                    ycci[key] = c[1][i][j]
                if all([v > 0 for v in ycci.values()]) and len(node_i.Xnames) > 1:
                    print('continuing')
                    Xnew = [x for x in node_i.Xnames if x != c[0]]
                    self.queue.append(Node(path, Xnew, ycci))
                else:
                    print('stopping')
                    print(path, ycci)
            self.nodes[self.nid] = node_i
            self.nid += 1
                
    
    
    def ContinGENcyTable(self, pathkeys, X):
        df_i = self.df
        print(pathkeys)
        for p in pathkeys:
            df_i = df_i.loc[df_i[p[0]] == p[2]]
        for x in X:
            xgroup = df_i.groupby(x).groups
            A = np.zeros((len(xgroup),len(self.ygroup)), dtype = np.int64)
            levels = list(xgroup.keys())
            for i, xc in enumerate(xgroup):
                for j, yc in enumerate(self.ygroup):
                    A[i][j] = np.sum(np.in1d(xgroup[xc],self.ygroup[yc],assume_unique=True))
            yield (x, A, levels)
    

            
            
            
#Decision tree data is held by intelligent nodes (that can also be leaves)                
class Node(object):
    
    def __init__(self, pathkeys, Xnames, yc_i, isleaf = False, ):
        self.isleaf = isleaf
        self.pathkeys = pathkeys
        self.Xnames = Xnames
        self.yc_i = yc_i
        self.opt = None
        self.levels = None
    
        