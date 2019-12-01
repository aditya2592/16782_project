import numpy as np
import os
import shutil
import argparse
import yaml
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from sklearn import mixture
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def create_3dwall(ax, x_range, y_range, z_range):
    # TODO: refactor this to use an iterator
    xx, yy = np.meshgrid(x_range, y_range)
    print(xx)
    ax.plot_wireframe(xx, yy, np.array([[z_range[0],z_range[0]]]), color="r", zorder=10)
    ax.plot_surface(xx, yy, np.array([[z_range[0],z_range[0]]]), color="r", alpha=0.5, zorder=10)
    ax.plot_wireframe(xx, yy, np.array([[z_range[1],z_range[1]]]), color="r", zorder=10)
    ax.plot_surface(xx, yy, np.array([[z_range[1],z_range[1]]]), color="r", alpha=0.5, zorder=10)


    yy, zz = np.meshgrid(y_range, z_range)
    ax.plot_wireframe(np.array([[x_range[0],x_range[0]]]), yy, zz, color="r", zorder=10)
    ax.plot_surface(np.array([[x_range[0],x_range[0]]]), yy, zz, color="r", alpha=0.5, zorder=10)
    ax.plot_wireframe(np.array([[x_range[1],x_range[1]]]), yy, zz, color="r", zorder=10)
    ax.plot_surface(np.array([[x_range[1],x_range[1]]]), yy, zz, color="r", alpha=0.5, zorder=10)

    xx, zz = np.meshgrid(x_range, z_range)
    ax.plot_wireframe(xx, np.array([[y_range[0],y_range[0]]]), zz, color="r", zorder=10)
    ax.plot_surface(xx, np.array([[y_range[0],y_range[0]]]), zz, color="r", alpha=0.5, zorder=10)
    ax.plot_wireframe(xx, np.array([[y_range[1],y_range[1]]]), zz, color="r", zorder=10)
    ax.plot_surface(xx, np.array([[y_range[1],y_range[1]]]), zz, color="r", alpha=0.5, zorder=10)

def generate_gaussian(samples, length, width, visualize=True, fig=None):

    X = np.arange(0, length, 0.1)
    Y = np.arange(0, width, 0.1)
    X_, Y_ = np.meshgrid(X, Y)
        
    g = mixture.GaussianMixture(n_components=6)  
    g.fit(samples)
    
    if visualize:
        Z_ = g.score_samples(np.concatenate((X_.reshape(-1,1), Y_.reshape((-1,1))), axis=1))
        Z_ = np.exp(Z_)
        Z_ = Z_.reshape(X_.shape)
        levels = np.arange(0, 1, 0.1)

        # fig = plt.figure()
        ax = fig.gca()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Sample Map')
        plt.xlim(0, length)
        plt.ylim(0, width)
        surf = ax.contourf(X_, Y_, Z_, levels,cmap=cm.Blues, zorder=-1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


def visualize(path_file, samples_file, length, width):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    samples = []
    with open(samples_file, "r") as f:
        line = f.readline().strip()
        while line:
            line = line.split(",")
            x = float(line[0])
            y = float(line[1])
            
            samples.append([x, y])
            line = f.readline()
    
    X = np.arange(0, length, 0.1)
    Y = np.arange(0, width, 0.1)
    # Z = np.zeros((X.shape[0], Y.shape[0]))
    
    X_, Y_ = np.meshgrid(X, Y)
        
    g = mixture.GaussianMixture(n_components=3)  
    g.fit(np.array(samples))
    
    Z_ = g.score_samples(np.concatenate((X_.reshape(-1,1), Y_.reshape((-1,1))), axis=1))
    
    # import pdb
    # pdb.set_trace()
    Z_ = np.exp(Z_)
    Z_ = Z_.reshape(X_.shape)
    
    levels = np.arange(0, 1, 0.1)
    # Plot the surface.
    surf = ax.plot_surface(X_, Y_, Z_, cmap=cm.Blues,
                        linewidth=0, antialiased=True, zorder=-1)
    # surf = ax.contourf(X_, Y_, Z_, levels,cmap=cm.Blues, zorder=-1)

    # Customize the z axis.
    ax.set_zlim(0, 1.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample Map')
    plt.xlim(0, length)
    plt.ylim(0, width)
    with open(path_file, "r") as f:
        line = f.readline()
        while line:
            line = line.split(" ")
            # print(line)
            if "wall" in line[0] in line[0]:
                x = float(line[1])
                y = float(line[2])
                z = float(line[3])
                l = float(line[4])
                b = float(line[5])
                h = float(line[6])
                # rect = Rectangle((x-l/2, y-b/2), l, b)
                create_3dwall(ax, np.array([x-l/2, x+l/2]),np.array([y-b/2, y+b/2]), np.array([z-h/2, z+h/2]))
                # ax.gca().add_patch(rect)
                
            line = f.readline()
    
    plt.draw()
    plt.show()
    
    ax.savefig("visualize.png")

def parse_arguments():
    '''
        parse commandline arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default=None,  help="map file path")
    parser.add_argument('--samples', dest='samples', type=str, default=None, help="sample file path")
    return parser.parse_args()

if __name__ == "__main__":
    '''
        entry point
    '''
    args = parse_arguments()
    
    map_file = args.map
    samples_file = args.samples
    
    with open('config.yaml') as f:
        config = yaml.load(f)
        
    length=config["map"]["x_max"]
    width=config["map"]["y_max"]
    
    visualize(map_file, samples_file, length, width)