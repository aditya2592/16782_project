import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_nearest_prob(arm, base, min_map, max_map, radius = 2, grid_res = 50):
    probs = []
    X = np.linspace(min_map,max_map,grid_res)
    pointsX, pointsY = np.meshgrid(X,X)
    points = np.vstack((pointsX.flatten(),pointsY.flatten())).T
    for i in range(points.shape[0]):
        a = 0
        b = 1
        p  = points[i,:]
        for j in range(1000):
            if (np.linalg.norm(np.subtract(base[j,:],p))<radius):
                b = b + 1
            if (np.linalg.norm(np.subtract(arm[j,:],p))<radius):
                a = a + 1
        probs.append(float(a)/(a+b))

    fig, ax = plt.subplots()
    scatter = ax.scatter(points[:,0], points[:,1], c=probs)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(scatter)
    plt.savefig('probs.png')