import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import random as rng
import pandas as pd
from statistics import mean


# Display one image
def display_one_plot(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()


def save_img(image, image_name, image_directory):
    if not os.path.isdir('processed_imgs/' + image_directory):
        os.mkdir('processed_imgs/' + image_directory)  # gelen parametre isminde klasör oluştu bunu varsa oluşturma
    img_tam_yolu = 'processed_imgs/' + image_directory + "/" + image_name + ".png"
    cv2.imwrite(img_tam_yolu, image)
    print(image_directory+"/"+image_name+"  saved\n")


def split_to_dirs(src_base_dir, dst_base_dir):
    if not os.path.isdir(dst_base_dir):
        os.mkdir(dst_base_dir)
    df_data = pd.read_csv('HAM10000_metadata.csv')
    df_data_bak = pd.read_csv('HAM10000_metadata.csv')

    df_data.set_index('image_id', inplace=True)
    image_list = list(df_data_bak['image_id'])

    ham_dir_list = os.listdir(src_base_dir)
    for image in image_list:
        img_name = image + '.jpg'
        label = df_data.loc[image, 'dx']
        print(label)
        if img_name in ham_dir_list:
            # source path to image
            src = os.path.join(src_base_dir, img_name)
            # destination path to image
            if not os.path.isdir(os.path.join(dst_base_dir, label)):
                os.mkdir(os.path.join(dst_base_dir, label))
            dst = os.path.join(dst_base_dir, label, img_name)
            # copy the image from the source to the destination
            shutil.copyfile(src, dst)


def load_img_list(src_base_dir):
    img_name_list = []
    img_label_list = []
    img_path_list = []

    dir_list = os.listdir(src_base_dir)
    for dir in dir_list:
        img_list = os.listdir(src_base_dir + "/" + dir)
        for img in img_list:
            img_name_list.append(img)
            img_label_list.append(dir)
            tmp_path = os.path.join(src_base_dir, dir)
            img_path = tmp_path + "/"+img
            img_path_list.append(img_path)
    return img_name_list, img_label_list, img_path_list


def load_img(img_path):
    img = cv2.imread(img_path)
    return img


def cvt_gray_img(img, color_code):
    cvt_img = cv2.cvtColor(img, color_code)
    return cvt_img


def improve_contrast_image_using_clahe(bgr_image):
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # clahe1 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    # clahe2 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    hsv_planes[0] = clahe.apply(hsv_planes[0])
    # hsv_planes[1] = clahe1.apply(hsv_planes[1])
    # hsv_planes[2] = clahe2.apply(hsv_planes[2])

    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def watershed_method(img):
    img = cvt_gray_img(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [0, 255, 0]
    dist2 = cv2.normalize(markers, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # display_one(markers,"markers")
    contours, hierarchy = cv2.findContours(dist2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(dist2, contours[0], -1, (0, 255, 0), 2)
    #
    # cv2.imshow("sss", cvt_gray_img(dist2, cv2.COLOR_GRAY2BGR))
    # cv2.imshow("counter", dist2)
    return img, sure_fg, contours


def extract_from_gray(img):
    # features metadata headers, returns a array of features

    all_gray_features = []
    src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(src_gray)
    blur = cv2.medianBlur(equ, 19, 11)
    ret, thresh = cv2.threshold(blur, 40, 255, 1)

    crop_img = thresh[65:400, 100:550]
    all_pixels = len(crop_img[0]) * len(crop_img[1])
    ones = cv2.countNonZero(crop_img)
    zeros = all_pixels - ones
    # print("nonZeroPix:{} ".format(ones), "zeroPix:{}".format(zeros))
    if ones > zeros:
        crop_img = 255 - crop_img
        print("image inverted")
        ones = cv2.countNonZero(crop_img)
        zeros = all_pixels - ones

    all_gray_features.append(ones)
    all_gray_features.append(zeros)

    # contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # bgr_crop_img = cvt_gray_img(crop_img, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        # Get the moments
        mu = cv2.moments(c)
        # Get the mass centers
        mc = (mu['m10'] / (mu['m00'] + 1e-5), mu['m01'] / (mu['m00'] + 1e-5))
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))

        cv2.drawContours(crop_img, [c], 0, color, 2)

        cont_cent_x = mc[0]
        cont_cent_y = mc[1]
        cv2.circle(crop_img, (int(cont_cent_x), int(cont_cent_y)), 4, color, -1)

        # cv2.imshow('contour and center', crop_img)
        # cv2.waitKey(0)

        cnt_area = cv2.contourArea(c)
        all_gray_features.append(cnt_area)           #

        cnt_arc_len = cv2.arcLength(c, True)
        all_gray_features.append(cnt_arc_len)        #

        x, y, w, h = cv2.boundingRect(c)
        cnt_rect_area = w * h
        all_gray_features.append(cnt_rect_area)       #

        cnt_aspect_ratio = float(w)/h
        all_gray_features.append(cnt_aspect_ratio)    #

        cnt_extend = float(cnt_area)/cnt_rect_area
        all_gray_features.append(cnt_extend)          #

        cnt_hull = cv2.convexHull(c)
        cnt_hull_area = cv2.contourArea(cnt_hull)
        cnt_solidity = float(cnt_area)/cnt_hull_area
        all_gray_features.append(cnt_solidity)        #

        cnt_equi_diameter = np.sqrt(4*cnt_area/np.pi)
        all_gray_features.append(cnt_equi_diameter)   #

        (x,y), (MA, ma), angle = cv2.fitEllipse(c)
        all_gray_features.append(MA)                  #
        all_gray_features.append(ma)                  #

    # cv2.drawContours(bgr_crop_img, contours, -1, (0, 255, 0), 3)
    resized_image = cv2.resize(crop_img, (7, 5), interpolation=cv2.INTER_LINEAR)
    img_pixel_data = resized_image.flatten()
    # print(type(img_pixel_data))
    # cv2.imshow("cropped", crop_img)
    # cv2.imshow("blur", blur)
    # cv2.imshow("fg", fg)
    # cv2.waitKey(0)
    return all_gray_features, img_pixel_data


def extract_from_color(img):
    # features metadata headers, returns a array of 380 element created from gray image
    # averageBlueColor  averageGreenColor    averageRedColor mostDominantColor1  mostDominantColor2   mostDominantColor3
    # mostDominantColor4   mostDominantColor5
    img = cv2.resize(img, (120, 90), cv2.INTER_LINEAR)
    avg_color_per_row = np.average(img, axis=0)
    avg_BGR_color = np.average(avg_color_per_row, axis=0)
    # print(avg_BGR_color.shape)
    #
    # pixels = np.float32(img.reshape(-1, 3))
    # n_colors = 3
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    # flags = cv2.KMEANS_RANDOM_CENTERS
    # _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    # _, counts = np.unique(labels, return_counts=True)
    # mostDominantColors = palette[np.argmax(counts)]
    # print(mostDominantColors.shape)
    # all_color_featuers = np.concatenate((avg_BGR_color, mostDominantColors), axis=0)
    # print(all_color_featuers.shape)

    return avg_BGR_color


def npy_dataset_concatenate(ds1, ds2):
    # for concatenating img_pixel_data, color_dataset, gray_dataset each other
    con = np.concatenate((ds1, ds2), axis=1)
    return con


def etl_for_all(src_base_dir):
    names, labels, paths = load_img_list(src_base_dir)
    dataset_gray_features = []
    dataset_color_features = []
    dataset_img_pixels = []

    for j in range(len(names)):
        img = load_img(paths[j])
        gray_features, img_pixel_data = extract_from_gray(img)
        color_features = extract_from_color(img)
        dataset_gray_features.append(gray_features)
        dataset_color_features.append(color_features)
        dataset_img_pixels.append(img_pixel_data)
        print(str(j)+"- img:{} processed.".format(names[j]))

    np_dataset_gray_features = np.array(dataset_gray_features)
    np_dataset_color_features = np.array(dataset_color_features)
    np_dataset_img_pixels = np.array(dataset_img_pixels)
    np_labels = np.array(labels)

    np.save("npy_data/gray_dataset.npy", np_dataset_gray_features)
    np.save("npy_data/color_dataset.npy", np_dataset_color_features)
    np.save("npy_data/img_pixel_dataset.npy", np_dataset_img_pixels)
    np.save("npy_data/label.npy", np_labels)


def etl_one_img(img):
    gray_features, img_pixel_data = extract_from_gray(img)
    color_features = extract_from_color(img)

    np_gray_features = np.array(gray_features)
    np_color_features = np.array(color_features)
    np_img_pixel_data = np.array(img_pixel_data)

    np_feature_list = np.append(np_gray_features, np_color_features)
    np_feature_list = np.reshape(np_feature_list, (1, 14))

    return np_feature_list, np_img_pixel_data

