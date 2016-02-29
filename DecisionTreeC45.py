import pandas as pd
import numpy as np
import collections
import webbrowser
import os

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
        self.queue.append(Node([], self.x, self.ycc, 'root'))
        
        while self.queue:
            
            node_i = self.queue.popleft()
            
            if len(node_i.Xnames) == 0 or any([v == 0 for v in node_i.yc_i.values()]):
                print('breaking')
                print(node_i.pathkeys, node_i.yc_i)
                node_i.isleaf = True
                node_i.decision = max(node_i.yc_i.items(), key = lambda x: x[1])[0]
                self.nodes[self.nid] = node_i
                self.nid += 1
                continue
            
            Info_y = I(node_i.yc_i.values())
            
            #print(node_i.Xnames)
            
            ct = self.ContinGENcyTable(node_i.pathkeys, node_i.Xnames)
            maxgain = 0
            
            T = sum(node_i.yc_i.values())
            for c in ct:
                print(c)
                Info_yx = np.sum(np.multiply(
                        np.apply_along_axis(I, 1, c[1]),
                        np.apply_along_axis(np.sum, 1, c[1]) / T
                        ))
                Gain_x = Info_y - Info_yx
                if Gain_x >= maxgain:
                    self.chosen = c
                    maxgain = Gain_x
            #print((self.chosen, maxgain))
            node_i.opt = self.chosen[0]
            print(node_i.opt)
            node_i.levels = self.chosen[2]
            Xnew = [x for x in node_i.Xnames if x != self.chosen[0]]
            print(Xnew)
            for i in range(len(node_i.levels)):
                path = list(node_i.pathkeys) + [(self.chosen[0],'=',self.chosen[2][i])]
                ycci = {}
                for j, key in enumerate(self.ygroup):
                    ycci[key] = self.chosen[1][i][j]
                self.queue.append(Node(path, Xnew, ycci, self.nid))
            self.nodes[self.nid] = node_i
            self.nid += 1
                
    
    
    def ContinGENcyTable(self, pathkeys, X):
        df_i = self.df
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
            
    def D3tree(self, filename, openpage=True):
        hier = {}
        for key in self.nodes:
            if self.nodes[key].isleaf:
                parent = self.nodes[key].decision
                hier[key] = {'name': parent, 'children': None}
            else:
                parent = self.nodes[key].opt
                d = [nid for nid, n in self.nodes.items() if n.parent == key]
                hier[key] = {'name': parent, 'children': d}
        higherkeys = list(hier.keys())
        higherkeys.reverse()
        for key in higherkeys:
            newlist = []
            if hier[key]['children']:
                for bbk in hier[key]['children']:
                    newlist.append(hier[bbk])
                    hier[key]['children'] = newlist
            else:
                del hier[key]['children']
        htmlf = 'TreeDiagrambase.html'
        red = open(htmlf, 'r').read()
        s = red % str(hier[0])
        htmlf = open(filename, 'w')
        htmlf.write(s)
        htmlf.close()
        if openpage:
            webbrowser.open_new_tab('file://'+os.getcwd()+'/'+filename)
    

            
            
            
#Decision tree data is held by intelligent nodes (that can also be leaves)                
class Node(object):
    
    def __init__(self, pathkeys, Xnames, yc_i, parent, isleaf = False):
        self.isleaf = isleaf
        self.decision = None
        self.pathkeys = pathkeys
        self.Xnames = Xnames
        self.yc_i = yc_i
        self.opt = None
        self.levels = None
        self.parent = parent
    
        