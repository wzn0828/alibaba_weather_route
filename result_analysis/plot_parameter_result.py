import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data_score = [(1, 11, 53632), (1, 12, 53754), (1, 13, 53290), (1, 14, 53176), (1, 15, 53060), (3, 14.5, 47024),
#               (4, 14.5, 46784), (3, 13, 49836), (3, 14, 47968), (3, 15, 46802), (4, 13, 48426), (4, 14, 46882),
#               (4, 15, 47484), (5, 14.5, 46714), (5, 15, 48062)]

data_score = [(3, 14.5, 47024), (4, 14.5, 46784), (3, 13, 49836), (3, 14, 47968), (3, 15, 46802), (4, 13, 48426),
              (4, 14, 46882),
              (4, 15, 47484), (5, 14.5, 46714), (5, 15, 48062)]

# data_craft_num = [(1, 11, 29), (1, 12, 29), (1, 13, 29), (1, 14, 29), (1, 15, 29), (3, 14.5, 25),
#               (4, 14.5, 25), (3, 13, 27), (3, 14, 26), (3, 15, 25), (4, 13, 26), (4, 14, 25),
#               (4, 15, 26), (5, 14.5, 25), (5, 15, 27)]
data_craft_num = [(3, 14.5, 25), (4, 14.5, 25), (3, 13, 27), (3, 14, 26), (3, 15, 25), (4, 13, 26), (4, 14, 25),
                  (4, 15, 26), (5, 14.5, 25), (5, 15, 27)]


def scatter():
    data = data_craft_num
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    z = [item[2] for item in data]
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()



def surface():
    data = data_score
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    z = [item[2] for item in data]

    x,y = np.meshgrid(x,y)

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1,cstride=1,cmap='rainbow')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


scatter()
# surface()