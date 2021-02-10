from operator import itemgetter

import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # dI/du Sobel filter
    filter_x = np.matrix([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # dI/dv Sobel filter
    filter_y = np.matrix([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    return filter_x, filter_y


def filter_image(im, filter):
    im_height, im_width = im.shape
    im_filtered = im.copy()
    im = np.pad(im, 1)
    filter_height, filter_width = filter.shape
    filter_width_half = int(filter_width / 2)
    filter_height_half = int(filter_height / 2)
    filter = filter.flatten().transpose()

    # iterate over the image and apply filter to each pixel
    for v in range(1, im_height + 1):
        for u in range(1, im_width + 1):
            # iterate over the window and apply the filter
            window = im[v - filter_height_half:v + filter_height_half + 1,
                     u - filter_width_half:u + filter_width_half + 1].reshape(-1)
            filtered_value = np.dot(window, filter)
            # set new image pixel data
            im_filtered[v - 1][u - 1] = filtered_value

    return im_filtered


def get_gradient(im_dx, im_dy):
    grad_mag = im_dx.copy()
    grad_angle = im_dx.copy()

    im_height, im_width = im_dx.shape

    for v in range(0, im_height):
        for u in range(0, im_width):
            dx = im_dx[v][u]
            dy = im_dy[v][u]
            # calculate magnitude and angle
            mag = np.sqrt(np.square(dx) + np.square(dy))
            angle = (np.arctan2(dy, dx))
            # set grad image data respectively
            grad_mag[v][u] = mag
            grad_angle[v][u] = angle

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    im_height, im_width = grad_mag.shape
    ori_histo = np.zeros((int((im_height - 1) / cell_size) + 1, int((im_width - 1) / cell_size) + 1, 6))

    for v in range(0, im_height):
        for u in range(0, im_width):
            # get current bin coordinates
            m = int(v / cell_size)
            n = int(u / cell_size)
            angle = (grad_angle[v][u])
            mag = grad_mag[v][u]

            # fill corresponding bin
            if (np.deg2rad(165) <= angle < np.deg2rad(180)) or (0 <= angle < np.deg2rad(15)):
                ori_histo[m][n][0] += mag
            elif np.deg2rad(15) <= angle < np.deg2rad(45):
                ori_histo[m][n][1] += mag
            elif np.deg2rad(45) <= angle < np.deg2rad(75):
                ori_histo[m][n][2] += mag
            elif np.deg2rad(75) <= angle < np.deg2rad(105):
                ori_histo[m][n][3] += mag
            elif np.deg2rad(105) <= angle < np.deg2rad(135):
                ori_histo[m][n][4] += mag
            elif np.deg2rad(135) <= angle < np.deg2rad(165):
                ori_histo[m][n][5] += mag

    return ori_histo


def get_block_descriptor(ori_histo, block_size=2):
    histo_height, histo_width, histo_depth = ori_histo.shape

    ori_depth = 6 * np.square(block_size)
    ori_width = histo_width - (block_size - 1)
    ori_height = histo_height - (block_size - 1)

    ori_histo_normalized = np.zeros((ori_height, ori_width, ori_depth))

    # fill ori_histo_normalized
    for v in range(0, ori_height):
        for u in range(0, ori_width):
            for y in range(0, block_size):
                for x in range(0, block_size):
                    for d in range(0, 6):
                        index = (y * block_size + x) * 6 + d
                        ori_histo_normalized[v][u][index] = ori_histo[v + y][u + x][d]

    # normalize
    for v in range(0, ori_height):
        for u in range(0, ori_width):
            square_sum = 0
            for d in range(ori_depth):
                square_sum += np.square(ori_histo_normalized[v][u][d])
            square_sum += np.square(0.0001)
            root_sum = np.sqrt(square_sum)
            for d in range(ori_depth):
                ori_histo_normalized[v][u][d] /= root_sum

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0

    # get differential Sobel filters
    filter_x, filter_y = get_differential_filter()

    # filter image
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)

    # get gradient magnitude and angle images
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)

    # get oriented histogram
    ori_histo = build_histogram(grad_mag, grad_angle, 8)

    # normalize oriented histogram
    ori_histo_normalized = get_block_descriptor(ori_histo, 2)
    if ori_histo_normalized.size > 0:
        hog = np.concatenate(ori_histo_normalized, axis=0)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int((im_h - 1) / cell_size) + 1, int((im_w - 1) / cell_size) + 1
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size ** 2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized ** 2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi / num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size * num_cell_w: cell_size],
                                 np.r_[cell_size: cell_size * num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=0.75, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    box_size = I_template.shape[0]

    target_height, target_width = I_target.shape

    template_hog = extract_hog(I_template)
    template_hog = template_hog.flatten()
    template_hog = (template_hog - np.mean(template_hog)) / (np.std(template_hog))
    template_hog_norm = np.linalg.norm(template_hog)

    target_hog = extract_hog(I_target)
    target_hog = target_hog.reshape((int((target_height - 1) / 8), int((target_width - 1) / 8), 24))

    bounding_boxes = []

    # calculate ncc for all applicable windows and append bounding boxes
    for v in range(0, target_height - box_size):
        for u in range(0, target_width - box_size):
            m = int(v / 8)
            n = int(u / 8)
            targ_hog = target_hog[m:m + 6, n:n + 6]
            targ_hog = (targ_hog - np.mean(targ_hog)) / np.std(targ_hog)
            target_hog_norm = np.linalg.norm(targ_hog)
            normalized_cross_correlation = (np.dot(template_hog, targ_hog.flatten()) / (
                        target_hog_norm * template_hog_norm) + 1) / 2
            if normalized_cross_correlation > 0.58:
                bounding_boxes.append([u, v, normalized_cross_correlation])

    bounding_boxes = np.array(bounding_boxes, list).tolist()

    # filter bounding boxes
    keep = []
    base_set = copy.copy(bounding_boxes)
    while True:
        if len(base_set) == 0:
            break
        new_set = []
        best_bb = max(base_set, key=itemgetter(2))
        for i in range(0, len(base_set)):
            current_bb = base_set[i]
            if intersection_of_union(best_bb, current_bb, box_size) < 0.4:
                new_set.append(current_bb)
        keep.append(best_bb)
        base_set = copy.copy(new_set)

    bounding_boxes = np.asarray(keep)

    return bounding_boxes


def intersection_of_union(bb1, bb2, box_size):
    bb1 = [bb1[0], bb1[1], bb1[0] + box_size, bb1[1] + box_size]
    bb2 = [bb2[0], bb2[1], bb2[0] + box_size, bb2[1] + box_size]
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def visualize_face_detection(I_target, bounding_boxes, box_size):
    hh, ww, cc = I_target.shape

    fimg = I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii, 0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1 < 0:
            x1 = 0
        if x1 > ww - 1:
            x1 = ww - 1
        if x2 < 0:
            x2 = 0
        if x2 > ww - 1:
            x2 = ww - 1
        if y1 < 0:
            y1 = 0
        if y1 > hh - 1:
            y1 = hh - 1
        if y2 < 0:
            y2 = 0
        if y2 > hh - 1:
            y2 = hh - 1
        fimg = cv2.rectangle(fimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f" % bounding_boxes[ii, 2], (int(x1) + 1, int(y1) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__ == '__main__':
    # MxN target image
    I_target = cv2.imread('target.png', 0)

    # MxN  face template
    I_template = cv2.imread('template.png', 0)

    # generate bounding boxes and filter out those that are correct
    bounding_boxes = face_recognition(I_target, I_template)

    # visualize bounding boxes
    I_target_c = cv2.imread('target.png')
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
