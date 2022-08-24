from msilib.schema import Error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from os import listdir
from os.path import join, exists, basename
import re

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

class FluentPoints:
    """
    This function plots a scatter plot using a plane and colours by the sand volume fraction
    it automatically excludes the plane that exhibits 0 std so that no manual specification is required
    df (pd dataframe): df from import_ascii
    exclude_plane (str): x,y or z that corresponds to the column name
    """
    def __init__(self,df,exclude_plane = None):
        self.df = df
        if exclude_plane is not None:
            self.exclude_plane = exclude_plane
        else:
            df_describe = df.describe()
            std_plane = df_describe.loc["std",:]
            axis = std_plane[std_plane==0.0] #zero std for a surface means that particular plane is being excluded
            axis = axis.index.values[0]
            if axis == 'x' or axis == 'y' or axis == 'z':
                self.exclude_plane = axis
            else:
                raise ValueError('Plane excluded is neither x,y nor z. Specify the exclude_plane argument')
        
    
    def get_points(self):
        df = self.df.drop(["nodenumber",self.exclude_plane],axis=1,inplace=False)
        s1 = df.iloc[:,0].values
        s2 = df.iloc[:,1].values
        c = df['sand'].values
        return s1,s2,c

    def plot_points(self):
        df = self.df.drop(["nodenumber",self.exclude_plane],axis=1,inplace=False)
        s1,s2,c = self.get_points()
        plt.figure()
        # plt.plot(x[::10],y[::10],'o')
        plt.scatter(s1,s2,c=c,marker='.')
        n1,n2 = df.iloc[:,:2].columns
        plt.xlabel(n1)
        plt.ylabel(n2)
        plt.show()
        return

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

def create_sorted_fp_list(directory,prefix):
    """
    directory (str): directory where all the ascii data is being stored
    prefix (str): name of the surface without the time-step (in ansys fluent, it should just be named as the surface e.g. xz_y48e-1)
    """
    fp_list = [f for f in listdir(directory) if f.startswith(prefix)]
    sorted_index = sorted([f.split('-')[1].zfill(5) for f in fp_list])
    sorted_fp_list = [join(directory,'{}-{}'.format(prefix,int(i))) for i in sorted_index]
    return sorted_fp_list


def create_tensor(fp_list,res1,res2):
    """
    create a multi-layer matrix (tensor), where the depth (last axis) is the surface at different time step
    fp_list (list of str): list of sorted filepaths so the tensor is in correct chronology order
    exclude_plane (str): x,y or z
    res1 (int): number of cells rows
    res2 (int): number of cells columns
    """
    # initialise the plane exclusion first so there is no need for repeated calculation for exclude_plane
    df = import_ascii(fp_list[0])
    flpt = FluentPoints(df)
    exclude_plane = flpt.exclude_plane #the same plane will be excluded from all the files with the same prefix, assuming the fp all consist of the same surface
    grid_tensor = np.zeros((res1,res2,len(fp_list)))
    for i,fp in enumerate(fp_list):
        df = import_ascii(fp)
        flpt = FluentPoints(df,exclude_plane=exclude_plane)
        s1,s2,c = flpt.get_points()
        _,_,zi = grid(s1,s2,c, res1=res1, res2=res2)
        grid_tensor[:,:,i] = zi
    return grid_tensor

def normalise_tensor(grid_tensor):
    """
    normalise the tensor such that the values fall between 0 and 1
    such that it represents radiometrically corrected images
    """
    min_v = grid_tensor.min()
    max_v = grid_tensor.max()
    normalisation = lambda x,min_v,max_v: (x - min_v)/(max_v-min_v)
    vectorised_norm = np.vectorize(normalisation)
    norm_grid = vectorised_norm(grid_tensor,min_v,max_v)
    return norm_grid

def save_tensor(t,fp_list,prefix,save_directory):
    """
    save the tensor with a unique prefix
    and create the metadata with the filepath name so we know where the tensor data comes from
    t (numpy array): tensor
    fp_list (list of str): list of sorted filepaths so the tensor is in correct chronology order
    prefix (str): file name (best to save it with the name of the surface e.g. xz_y48e-1)
    save_directory (str): directory where u want to save the data
    """
    # save metadata of the fp list
    metadata_fp = join(save_directory,'{}_metadata_fp_list.txt'.format(prefix))
    if exists(metadata_fp) is True:
        raise NameError('file already exists! Provide a unique prefix!')
    else:
        with open(metadata_fp, 'w') as fp:
            for item in fp_list:
                # write each item on a new line
                fp.write("%s\n" % item)
    # save tensor
    fp = join(save_directory,"{}_res1{}_res2{}_depth{}".format(prefix,t.shape[0],t.shape[1],t.shape[2]))
    if exists(fp) is True:
        raise NameError('file already exists! Provide a unique prefix!')
    else:
        t.tofile(fp)
    return

def load_tensor(fp):
    """
    fp (str): file path of the saved tensor
    """
    res1,res2,depth = fp_name = basename(fp).split('_')[-3:]
    res1 = int(res1.replace('res1',''))
    res2 = int(res2.replace('res2',''))
    depth = int(depth.replace('depth',''))
    array = np.fromfile(fp,dtype= float)
    print("Load fp: {}".format(fp))
    array = array.reshape((res1,res2,depth))
    print("array shape: {}".format(array.shape))
    return array




