import numpy as np
import matplotlib.pyplot as plt

def visualize(x, width = -1, height = -1, pad = 5, title = 'FeaturePlot.png'):    
    size = x.shape
    if width is -1 or height is -1:    
        width = np.sqrt(size[1])
        height = width
    
    denom =  np.dot(np.sqrt(np.sum(x**2,axis=1)).reshape(-1,1),np.ones((1,x.shape[1])))
    x = x/denom
    grid = -1*np.ones((pad+5*(width+pad), pad+5*(pad+height)))
    for i in range(5):
        for j in range(5):
            if i*5+j == x.shape[0]:
                break
            grid[((i+1)*pad+i*width):((i+1)*(pad+width)),((j+1)*pad+j*height):((j+1)*(pad+height))] = x[i*5+j,:].reshape(width, height)/np.max(np.abs(x[i*5+j,:]))
        if i*5+j == x.shape[0]:
            break
    f = plt.figure()
    plt.imshow(grid, cmap='Greys')
    plt.show()
    f.savefig(title)