import pandas as pd
import numpy as np
import collections
import webbrowser
import os

def I(counts):
        i = 0
        total = sum(counts)
        for c in counts:
            if c:
                i += -(c/total)*np.log2(c/total)
            else:
                i += 0
        return(i)

class DecisionTree(object):
    
    #Instances of DecisionTree objects must initialize with a pointer to a pandas dataframe
    def __init__(self, df):
        self.df = df
        self.isfit = False
    
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
            
            if node_i.parent != 'root':
                self.nodes[node_i.parent].children.append(self.nid)
            
            if len(node_i.Xnames) == 0 or len([v for v in node_i.yc_i.values() if v > 0]) < 2:
                node_i.isleaf = True
                node_i.decision = max(node_i.yc_i.items(), key = lambda x: x[1])[0]
                self.nodes[self.nid] = node_i
                self.nid += 1
                continue
            
            Info_y = I(node_i.yc_i.values())
            
            ct = self.ContingencyTable(node_i.pathkeys, node_i.Xnames)
            
            maxgain = 0
            
            T = sum(node_i.yc_i.values())
            
            for c in ct:

                Info_yx = np.sum(np.multiply(
                        np.apply_along_axis(I, 1, c[1]),
                        np.apply_along_axis(np.sum, 1, c[1]) / T
                        ))
                Gain_x = Info_y - Info_yx
                if Gain_x > maxgain:
                    chosen = c
                    maxgain = Gain_x
            
            if maxgain == 0:
                node_i.isleaf = True
                node_i.decision = max(node_i.yc_i.items(), key = lambda x: x[1])[0]
                self.nodes[self.nid] = node_i
                self.nid += 1
                continue

            node_i.opt = chosen[0]

            node_i.levels = chosen[2]
            Xnew = [x for x in node_i.Xnames if x != chosen[0]]

            for i in range(len(node_i.levels)):
                path = list(node_i.pathkeys) + [(chosen[0],'=',chosen[2][i])]
                ycci = {}
                for j, key in enumerate(self.ygroup):
                    ycci[key] = chosen[1][i][j]
                self.queue.append(Node(path, Xnew, ycci, self.nid))
            self.nodes[self.nid] = node_i
            self.nid += 1
        
        self.isfit = True
        
    def ContingencyTable(self, pathkeys, X):
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
            
    def autoprune(self):
        if not self.isfit:
            print("ERROR: NO TREE TO AUTOPRUNE")
            return(None)
        for key in reversed(list(self.nodes.keys())):
            if not self.nodes[key].isleaf:
                dlist = [self.nodes[ckey].decision for ckey in self.nodes[key].children]
                dset = set(dlist)
                if len(dset) == 1 and dset != {None}:
                    self.nodes[key].isleaf = True
                    self.nodes[key].decision = dlist[0]
                    for ckey in self.nodes[key].children:
                        del self.nodes[ckey]
                    self.nodes[key].children = None
        
    
    def D3tree(self, filename, openpage=True):
        if not self.isfit:
            print("ERROR: NO TREE TO CHART")
            return(None)
        js = self.Nodes2JS()
        treebase = open('TreeDiagrambase.html', 'r')
        red = treebase.read()
        s = red % str(js)
        htmlf = open(filename, 'w')
        htmlf.write(s)
        treebase.close()
        htmlf.close()
        if openpage:
            webbrowser.open_new_tab('file://'+os.getcwd()+'/'+filename)
    
    def Nodes2JS(self):
        hier = {}
        for key in self.nodes:
            if self.nodes[key].isleaf:
                cat = ''
                if self.nodes[key].pathkeys:
                    cat = self.nodes[key].pathkeys[-1][-1]
                hier[key] = {'name': self.nodes[key].decision,
                             'counts' : ((str(self.nodes[key].yc_i)).replace('{','')).replace('}',''),
                             'category': cat,
                             'leaf': 'true',
                             'children': None}
            else:
                cat = ''
                if self.nodes[key].pathkeys:
                    cat = self.nodes[key].pathkeys[-1][-1]
                hier[key] = {'name': self.nodes[key].opt,
                             'counts' : ((str(self.nodes[key].yc_i)).replace('{','')).replace('}',''),
                             'category': cat,
                             'leaf': 'false',
                             'children': self.nodes[key].children}
        for key in reversed(list(hier.keys())):
            newlist = []
            if hier[key]['children']:
                for bbk in hier[key]['children']:
                    newlist.append(hier[bbk])
                    del hier[bbk]
                hier[key]['children'] = newlist
            else:
                del hier[key]['children']
        return(hier[0])
        
            
            
            
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
        self.children = []
    
        