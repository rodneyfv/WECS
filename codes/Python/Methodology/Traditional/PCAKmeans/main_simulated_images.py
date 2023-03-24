# import gdal
from osgeo import gdal
import numpy as np
from Methodology.Traditional.PCAKmeans.algorithm import pca_k_means
from Methodology.Traditional.PCAKmeans.util import diff_image
from Methodology.util.cluster_util import otsu
import imageio.v3
import time


def main():
    before_img0 = gdal.Open('../../../Dataset/simulated_image_t1.tiff')  # data set X
    after_img0 = gdal.Open('../../../Dataset/simulated_image_t80.tiff')  # data set Y

    f = open('details-PCAKA_simulated_images.txt', 'w')
    print("Here we provide information about the analysis of the simulated images using the PCA-Kmeans method\n", file=f)
    # before_img = np.sqrt(np.power(before_img[0, :, :], 2) + np.power(before_img[1, :, :], 2))
    img_width = before_img0.RasterXSize  # image width
    img_height = before_img0.RasterYSize  # image height

    before_img = np.zeros([img_height, img_width, 2])
    tmp = before_img0.GetRasterBand(1)
    before_img[:, :, 0] = tmp.ReadAsArray()
    tmp = before_img0.GetRasterBand(2)
    before_img[:, :, 1] = tmp.ReadAsArray()

    after_img = np.zeros([img_height, img_width, 2])
    tmp = after_img0.GetRasterBand(1)
    after_img[:, :, 0] = tmp.ReadAsArray()
    tmp = after_img0.GetRasterBand(2)
    after_img[:, :, 1] = tmp.ReadAsArray()

    del before_img0, after_img0, tmp

    # before_img = imageio.imread('../../../Dataset/PCAKmeans/burn_1986.png')[:, :, 0:3]
    # after_img = imageio.imread('../../../Dataset/PCAKmeans/burn_1992.png')[:, :, 0:3]
    # print(type(before_img))
    # print(before_img.shape)

    eig_dim = 10*2
    block_sz = 4*2

    tic = time.time()

    diff_img = diff_image(before_img, after_img, is_abs=True, is_multi_channel=True)
    del before_img, after_img
    change_img = pca_k_means(diff_img, block_size=block_sz,
                             eig_space_dim=eig_dim)
    del diff_img
    imageio.imwrite('PCAKmeans_changeimage_simulated_images.png', change_img)

    bcm = np.ones((img_height, img_width))
    thre = otsu(change_img)
    bcm[change_img > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    imageio.imwrite('PCAKmeans_simulated_images.tiff', bcm)
    imageio.imwrite('PCAKmeans_simulated_images.png', bcm)

    toc = time.time()
    print("The analysis used Otsu's threshold", file=f)
    print("Threshold: ", thre, file=f)
    print("Time to run: ", round(toc - tic, 4), "s", file=f)
    f.close()



if __name__ == '__main__':
    main()
