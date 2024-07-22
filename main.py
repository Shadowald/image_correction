import cv2
import numpy as np
import glob
import timeit
import time

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    # Collect images in a list for later
    imgs = glob.glob('./vignette/vignette_*.bmp')
    imgs_low_res = glob.glob('./vignette/low_res_*.bmp')

    # Make list to hold fixed images later
    imgs_fix = []
    hsv_fix = []
    hsv_fix2 = []
    hsv_fix3 = []

    # Get Calibration images and extract dimensions
    calibration_img = cv2.imread('./vignette/vignette_weiss.bmp')
    calibration_low_img = cv2.imread('./vignette/low_res_weiss.bmp')
    calibration_hsv = cv2.cvtColor(calibration_img, cv2.COLOR_RGB2HSV)
    height, width, depth = cv2.imread(imgs[0]).shape
    height_low, width_low, depth_low = cv2.imread(imgs_low_res[0]).shape

    # Write image for debugging
    cv2.imwrite('./vignette/hsv_weiss.bmp', calibration_hsv)

    # Set coodinates to make croppped image
    crop_height_start = int(height / 4)
    crop_height_end = int(height - crop_height_start)
    crop_width_start = int(width / 4)
    crop_width_end = int(width - crop_width_start)

    crop_height_low_start = int(height_low / 4)
    crop_height_low_end = int(height_low - crop_height_low_start)
    crop_width_low_start = int(width_low / 4)
    crop_width_low_end = int(width_low - crop_width_low_start)

    # Create the cropped images
    crop_img = calibration_img[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
    crop_hsv = calibration_hsv[crop_height_start:crop_height_end, crop_width_start:crop_width_end]
    crop_low_img = calibration_low_img[crop_height_low_start:crop_height_low_end, crop_width_low_start:crop_width_low_end]
    # cv2.imwrite('./vignette/crop_image.bmp', crop_img)


    # Split HSV values
    h_crop, s_crop, v_crop = cv2.split(crop_hsv)
    h, s, v = cv2.split(calibration_hsv)
    # cv2.imshow("v channel", v)
    # cv2.waitKey()

    # Find the average value of the cropped images
    # avg = np.average(crop_img)
    avg = cv2.mean(crop_img)
    avg_hsv = cv2.mean(crop_hsv)
    v_avg = np.average(v_crop)
    avg_low = np.average(crop_low_img)

    # Calculate correction V value
    correct_v = v / v_avg
    correct_v2 = v_avg / v
    normalize_v = correct_v2
    cv2.normalize(correct_v2, normalize_v, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imshow("correct_v channel", normalize_v)
    # cv2.waitKey()

    # Merge HSV values
    h.fill(1)
    s.fill(1)
    correct_v_type = np.clip(correct_v, 0, 255).astype(crop_hsv.dtype)
    correct_hsv = cv2.merge((h, s, correct_v_type))
    correct_v_type2 = np.clip(correct_v2, 0, 255).astype(crop_hsv.dtype)
    correct_hsv2 = cv2.merge((h, s, correct_v_type2))
    # correct_hsv = cv2.resize(correct_hsv, (width, height))

    # Make a white image to create the correction image
    wht_img = np.zeros([height, width, depth], np.float32)
    wht_img.fill(255)
    wht_low_img = np.zeros([height_low, width_low, depth_low], np.float32)
    wht_low_img.fill(255)

    # Make the correction image
    correction_img = np.divide(wht_img, calibration_img)
    correction_low_img = np.divide(wht_low_img, calibration_low_img)
    # cv::divide(wht_img, calibration_img, correction_img)

    # Loop over all images to correct
    for i in range(len(imgs)):
        # Correct images using correction image
        img = cv2.imread(imgs[i])
        imgs_fix.append(np.multiply(img, correction_img))

        # Correct image by changing the V value in the HSV image to averaged V of the cropped HSV image
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv2 = img_hsv
        img_hsv3_1 = img_hsv
        img_hsv3_2 = img_hsv
        h, s, v = cv2.split(img_hsv)

        # Correction using v / avg_v, just modifying v
        # correct_v = cv2.resize(correct_v, (width, height))
        v1 = v / correct_v
        v2 = v * correct_v2
        v1 = np.clip(v1, 0, 255).astype(img_hsv.dtype)
        img_hsv = cv2.merge((h, s, v1))
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        # Correction using avg_v / v, just modifying v
        # correct_v2 = cv2.resize(correct_v2, (width, height))
        v2 = v * correct_v2
        v2 = np.clip(v2, 0, 255).astype(img_hsv2.dtype)
        img_hsv2 = cv2.merge((h, s, v2))
        img_hsv2 = cv2.cvtColor(img_hsv2, cv2.COLOR_HSV2RGB)

        # Correction using v / avg_v using corrected image
        img_hsv3_1 = np.multiply(img_hsv3_1, correct_hsv)
        img_hsv3_2 = np.multiply(img_hsv3_2, correct_hsv2)
        img_hsv3 = cv2.add(img_hsv3_1, img_hsv3_2)

        # Append images to list to later be written
        hsv_fix.append(img_hsv)
        hsv_fix2.append(img_hsv2)
        hsv_fix3.append(cv2.cvtColor(img_hsv3, cv2.COLOR_HSV2RGB))
        # imgs_fix[i] = img.mul(correction_img)

    # Write the corrected images for viewing
    cv2.imwrite('./vignette/correction_image.bmp', correction_img)
    cv2.imwrite('./vignette/gray_card_fix.bmp', imgs_fix[0])
    cv2.imwrite('./vignette/schalen_leer_fix.bmp', imgs_fix[1])
    cv2.imwrite('./vignette/schalen_testobjekte_fix.bmp', imgs_fix[2])
    cv2.imwrite('./vignette/weiss_weiss.bmp', imgs_fix[3])
    cv2.imwrite('./vignette/gray_card_hsv.bmp', hsv_fix[0])
    cv2.imwrite('./vignette/schalen_leer_hsv.bmp', hsv_fix[1])
    cv2.imwrite('./vignette/schalen_testobjekte_hsv.bmp', hsv_fix[2])
    cv2.imwrite('./vignette/weiss_hsv.bmp', hsv_fix[3])
    cv2.imwrite('./vignette/gray_card_hsv2.bmp', hsv_fix2[0])
    cv2.imwrite('./vignette/schalen_leer_hsv2.bmp', hsv_fix2[1])
    cv2.imwrite('./vignette/schalen_testobjekte_hsv2.bmp', hsv_fix2[2])
    cv2.imwrite('./vignette/weiss_hsv2.bmp', hsv_fix2[3])
    cv2.imwrite('./vignette/gray_card_hsv3.bmp', hsv_fix3[0])
    cv2.imwrite('./vignette/schalen_leer_hsv3.bmp', hsv_fix3[1])
    cv2.imwrite('./vignette/schalen_testobjekte_hsv3.bmp', hsv_fix3[2])
    cv2.imwrite('./vignette/weiss_hsv3.bmp', hsv_fix3[3])

    # Create new "white" image based on average of the cropped image
    # wht_img.fill(avg)
    wht_img[:] = (avg[0], avg[1], avg[2])
    cv2.imwrite('./vignette/wht_img.bmp', wht_img)

    # Create new correction image
    correction_img = np.divide(wht_img, calibration_img)
    # correction_img = cv2.divide(wht_img, calibration_img)
    # cv::divide(wht_img, calibration_img, correction_img)

    # Correct all oimages again using new correction image
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i])
        imgs_fix[i] = np.multiply(img, correction_img)
        # imgs_fix[i] = img.mul(correction_img)

    # Write new corrected images for viewing
    cv2.imwrite('./vignette/correction_image2.bmp', correction_img)
    cv2.imwrite('./vignette/gray_card_fix2.bmp', imgs_fix[0])
    cv2.imwrite('./vignette/schalen_leer_fix2.bmp', imgs_fix[1])
    cv2.imwrite('./vignette/schalen_testobjekte_fix2.bmp', imgs_fix[2])
    cv2.imwrite('./vignette/weiss_weiss2.bmp', imgs_fix[3])

    # Create new containers to hold an image for time testing purposes
    img = cv2.imread(imgs[0])
    img_low = cv2.imread(imgs_low_res[0])

    # Time how long the multiplication takes for a single operation and for 1000 operations for both a high res
    # and low res image
    tic = time.perf_counter()
    np.multiply(img, correction_img)
    toc = time.perf_counter()
    print(toc - tic)

    tic = time.perf_counter()
    for i in range(10):
        np.multiply(img, correction_img)
    toc = time.perf_counter()
    print(toc - tic)

    tic = time.perf_counter()
    np.multiply(img_low, correction_low_img)
    toc = time.perf_counter()
    print(toc - tic)

    tic = time.perf_counter()
    for i in range(10):
        np.multiply(img_low, correction_low_img)
    toc = time.perf_counter()
    print(toc - tic)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tic = time.perf_counter()
    h, s, v = cv2.split(img_hsv)
    v = v / correct_v
    v = np.clip(v, 0, 255).astype(img_hsv.dtype)
    cv2.merge([h, s, v])
    toc = time.perf_counter()
    print(toc - tic)

    tic = time.perf_counter()
    for i in range(10):
        h, s, v = cv2.split(img_hsv)
        v = v / correct_v
        v = np.clip(v, 0, 255).astype(img_hsv.dtype)
        cv2.merge([h, s, v])
    toc = time.perf_counter()
    print(toc - tic)

    # tic = time.perf_counter()
    # np.multiply(img_hsv, correct_hsv)
    # toc = time.perf_counter()
    # print(toc - tic)

    # tic = time.perf_counter()
    # for i in range(1000):
    #     np.multiply(img_hsv, correct_hsv)
    # toc = time.perf_counter()
    # print(toc - tic)


def correct_img(img, correction_img):
    return np.multiply(img, correction_img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
