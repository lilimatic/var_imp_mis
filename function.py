import pandas as pd
import numpy as np
from matplotlib.patches import PathPatch

class dataset:
    def __init__(self,n,df):
        #Number of observations
        self.n = n
        self.df = df   
    
    def ishigami(self,eps):
        return lambda a, b:  np.sin(self.df['X1']) + a * np.sin(self.df['X2'])**2 + b * (self.df['X3'])**4 * np.sin(self.df['X1']) + eps
    
    def picked(self,request,a,b):
        #request are the variables to be randomized, e.g. ['X2','X3']
        newdf = self.df.copy()
        values  = [list(pd.Series(np.random.uniform(0,1,self.n)))]*len(request)
        dictionary = dict(zip(request, values))
        for col, new_values in dictionary.items():
            newdf = newdf.assign(**{col: new_values})
        return newdf
    

#Logistic probability of observation 
def pi(y,b0,b1):
    lin = b0 + b1 *y 
    return 1/(1+np.exp(-lin))

def sobol(df,n1,request,sim,a,b,eps):
    sobol_list = []
    df1 = df.copy()
    for x in range(sim):
        df_pf   = dataset(n1,df).picked(request,a,b)
        df1['Y']    =dataset(n1,df1).ishigami(eps)(a,b)
        df_pf['Y'] = dataset(n1,df_pf).ishigami(eps)(a,b)
        sobol_list.append((np.cov(df1.Y,df_pf.Y)/np.var(df1.Y))[0][1])
    return sobol_list

def singletons(df,n1,sim,a,b,eps):
    singleton = []
    for x in [['X2','X3'],['X1','X3'],['X1','X2']]:
        singleton.append(sobol(df,n1,x,sim,a,b,eps))
    return pd.DataFrame(dict(zip(['X1','X2','X3'],singleton)))



def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])
                        
                        
