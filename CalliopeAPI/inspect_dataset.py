import numpy as np
from matplotlib import pyplot as plt
import sys


def main():
    if len(sys.argv) != 1:
        try:
            img = np.load(sys.argv[1], allow_pickle=True)
            for i in img:
                plt.imshow(i)
                plt.show()
        except IOError:
            print("Invalid file!")

    else:
        print('Please specify dataset location as command argument!')


if __name__=='__main__':
    main()
