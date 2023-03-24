import numpy as np
import imageio
# import gdal
from osgeo import gdal
import time
from Methodology.util.cluster_util import otsu
from Methodology.util.data_prepro import stad_img
import matplotlib.pyplot as plt


def CVA(img_X, img_Y, stad=False):
    # CVA has not affinity transformation consistency, so it is necessary to normalize multi-temporal images to
    # eliminate the radiometric inconsistency between them
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm


def main():
    data_set_X = gdal.Open('../../../Dataset/simulated_image_t1.tiff')  # data set X
    data_set_Y = gdal.Open('../../../Dataset/simulated_image_t80.tiff')  # data set Y

    # file to write details of the analysis
    f = open('details-simulated_images.txt', 'w')
    print("Here we provide information about the analysis of the first "
          "and last simulated images using the CVA method\n", file=f)
    # print(type(data_set_X))
    # print(data_set_X.RasterCount)


    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    # plt.imshow(img_X[0, :, :])
    # plt.show()
    # plt.imshow(img_X[1, :, :])
    # plt.show()
    # exit()

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y)

    bcm = np.ones((img_height, img_width))
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    print(type(bcm))
    print(bcm.shape)
    imageio.imwrite('CVA_simul_im.tiff', bcm)
    imageio.imwrite('CVA_simul_im.png', bcm)

    # plt.imshow(bcm)
    # plt.show()
    # exit()

    toc = time.time()
    print("The analysis used Otsu's threshold", file=f)
    print("Threshold: ", thre, file=f)
    print("Time to run: ", round(toc - tic, 4), "s", file=f)
    f.close()



if __name__ == '__main__':
    main()
