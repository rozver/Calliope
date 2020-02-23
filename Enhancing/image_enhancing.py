from PIL import Image
import imageio
import numpy as np

def impro(img):
    img_out = img
    height = img.shape[0]
    width = img.shape[1]

    for i in np.arange(3, height - 3):
        for j in np.arange(3, width - 3):
            neighbors = []
            for k in np.arange(-3, 4):
                for l in np.arange(-3, 4):
                    neighbors.append((img.item(i + k, j + l, 0), img.item(i + k, j + l, 1), img.item(i + k, j + l, 2)))
            neighbors.sort()

            averaging = []

            for k in range(1, len(neighbors) - 1):
                if (abs(neighbors[k][0] - neighbors[k - 1][0]) < 10 and abs(neighbors[k][1] - neighbors[k - 1][1]) < 10 and abs(neighbors[k][1] - neighbors[k - 1][1]) < 10) or (abs(neighbors[k][0] - neighbors[k + 1][0]) < 10 and abs(neighbors[k][1] - neighbors[k + 1][1]) < 10 and abs(neighbors[k][1] - neighbors[k + 1][1]) < 10):
                    averaging.append(neighbors[k])
            
            r, g, b = 0, 0, 0
            lenght = len(averaging)
            for k in range(lenght):
                r += averaging[k][0]
                g += averaging[k][1]
                b += averaging[k][2]

            r /= lenght
            g /= lenght
            b /= lenght

            b = (r, g, b)
            img_out.itemset((i,j,0), b[0])
            img_out.itemset((i,j,1), b[1])
            img_out.itemset((i,j,2), b[2])

    return img_out

def median(img):
    img_out = img
    height = img.shape[0]
    width = img.shape[1]

    for i in np.arange(3, height-3):
        for j in np.arange(3, width-3):
            neighbors = []
            for k in np.arange(-3, 4):
                for l in np.arange(-3, 4):
                    a = (img.item(i+k, j+l, 0), img.item(i+k, j+l, 1), img.item(i+k, j+l, 2))
                    neighbors.append(a)
            neighbors.sort()
            median = neighbors[24]
            b = median
            img_out.itemset((i,j,0), b[0])
            img_out.itemset((i,j,1), b[1])
            img_out.itemset((i,j,2), b[2])
    return img_out

data = imageio.imread('D:/Programs/Solutions/My Projects/Calliope/Enhancing/images/balloons_noisy.png')

data = impro(data)
outimage = Image.fromarray(impro(data), 'RGB')
outimage.save('D:/Programs/Solutions/My Projects/Calliope/Enhancing/images/median_balloons.png')
outimage.show()