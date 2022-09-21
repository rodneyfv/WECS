import numpy as np
import imageio
# import gdal
from osgeo import gdal
import time
from Methodology.util.cluster_util import otsu
from Methodology.util.data_prepro import stad_img
# import matplotlib.pyplot as plt


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
    data_set_X = gdal.Open('../../../Dataset/2015-12-26.tif')  # data set X
    data_set_Y = gdal.Open('../../../Dataset/2017-12-3.tif')  # data set Y

    # file to write details of the analysis
    f = open('details-CVA_BrazilGuiana.txt', 'w')
    print("Here we provide information about the analysis of the Brazil-Guiana data using the CVA method\n", file=f)
    # print(type(data_set_X))
    # print(data_set_X.RasterCount)

    # band_1 = data_set_X.GetRasterBand(1)  # red channel
    # b1 = band_1.ReadAsArray()
    # plt.imshow(b1)
    # plt.savefig('Tiff.png')
    # plt.show()
    # band_1 = data_set_Y.GetRasterBand(1)  # red channel
    # b1 = band_1.ReadAsArray()
    # plt.imshow(b1)
    # plt.savefig('Tiff2.png')

    # data_set_X = gdal.Open('../../../Dataset/Landsat/Taizhou/2000TM')  # data set X
    # data_set_Y = gdal.Open('../../../Dataset/Landsat/Taizhou/2003TM')  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y)

    bcm = np.ones((img_height, img_width))
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    print(type(bcm))
    print(bcm.shape)
    imageio.imwrite('CVA_BrazilGuiana2.tiff', bcm)
    imageio.imwrite('CVA_BrazilGuiana.png', bcm)
    toc = time.time()
    print("The analysis used Otsu's threshold", file=f)
    print("Threshold: ", thre, file=f)
    print("Time to run: ", round(toc - tic, 4), "s", file=f)
    f.close()



if __name__ == '__main__':
    main()
