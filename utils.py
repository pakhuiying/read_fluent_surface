import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata

def import_ascii(fp):
    """
    This function converts ascii data to a pandas dataframe
    fp (str): filepath of ascii data (in fluent export the dsta as ascii)
    columns: 'nodenumber','x','y','z','water','sand', where water and sand represents the volume fraction of water and sand
    """
    x = np.genfromtxt(fp, dtype=None)
    df = pd.DataFrame(x,columns=['nodenumber','x','y','z','water','sand'])
    df = df.iloc[1:,:]
    df = df.astype(float)
    # print(df.head())
    return df

def plot_points(df,exclude_plane="z",show_plot=True):
    """
    This function plots a scatter plot using a plane and colours by the sand volume fraction
    df (pd dataframe): df from import_ascii
    exclude_plane (str): x,y or z that corresponds to the column name
    """
    # convert x,y coordinates to points on a mesh
    df = df.drop(["nodenumber",exclude_plane],axis=1,inplace=False)
    # df['sand'] = df['sand']*255
    # print(df['sand'].dtype)
    s1 = df.iloc[:,0].values
    s2 = df.iloc[:,1].values
    c = df['sand'].values
    if show_plot is True:
        plt.figure()
        # plt.plot(x[::10],y[::10],'o')
        plt.scatter(s1,s2,c=c,marker='.')
        n1,n2 = df.iloc[:,:2].columns
        plt.xlabel(n1)
        plt.ylabel(n2)
        plt.show()
    return s1,s2,c

def grid(s1,s2,c, res1=1000, res2=1000):
    """
    This function interpolates scatter points with irregular grid size to a grid (matrix)
    s1,s2,c (numpy arrays): output from plot_points
    res1, res2 (int): number of rows and columns in a mesh grid
    >>> xi,yi,zi = grid()
    where zi can be plotted using plt.imshow
    """
    # define the grid
    xi = np.linspace(min(s1), max(s1), res1)
    yi = np.linspace(min(s2), max(s2), res2)
    # interpolate
    zi = griddata((s1,s2), c, (xi[None,:], yi[:,None]),method='linear')
    return xi,yi,zi

def plot_contour(s1,s2,c,res=15):
    """
    This function plots the contour map of the volume fraction of sand using just the scatter points
    s1,s2,c (numpy arrays): output from plot_points
    res (int): determine how many contours are created. Higher value means more contours are created
    """
    plt.figure()
    plt.tricontour(s1,s2,c, res, linewidths=0.5, colors='k')
    plt.tricontourf(s1,s2,c, res)

def plot_grid(zi,cmap="BrBG_r"):
    """
    This function plots the grid interpolated from the grid() function
    """
    plt.figure()
    plt.imshow(zi,cmap=cmap)
    plt.colorbar()
    return

