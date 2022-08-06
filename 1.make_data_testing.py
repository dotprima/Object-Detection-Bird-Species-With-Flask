import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib.image as mpimg
import os

# Praproses - Gausian Noise
sdir = 'newtest/'
file_name = ''+sdir
resp = {}
#buat folder
if os.path.exists(file_name) == False:
    os.mkdir(''+sdir)

dir = "test/"
list_class = os.listdir(dir) 
total_class = len(list_class)
for i in range(total_class):

    folder = ""+sdir+list_class[i]
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    list_data = os.listdir(dir+list_class[i]) 
    total_data = len(list_data)
    
    #loop data file in class
    for j in range(total_data):
        name = list_data[j]
        lokasi_save = folder+'/'+'1.jpg'
        lokasi_file = dir+list_class[i]+'/'+list_data[j]
        resp[j] = lokasi_file

    im1 = mpimg.imread(resp[0])
    im2 = mpimg.imread(resp[1])
    im3 = mpimg.imread(resp[2])
    im4 = mpimg.imread(resp[3])
    im5 = mpimg.imread(resp[4])

    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, 3),  # creates 2x2 grid of axes
                    axes_pad=0.4,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, [im1, im2, im3, im4,im5]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(lokasi_save) 
    plt.clf()   

    
        