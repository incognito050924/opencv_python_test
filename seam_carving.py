import cv2
import numpy as np
import sys

# Draw vertical seam on top of the image.
def overlay_verical_seam(img, seam):
    img_seam_overlay = img.copy()

    # Extract the list of points from the seam.
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])

    # Draw a green line on the image using the list of points
    img_seam_overlay[x_coords, y_coords] = (0, 255, 0)
    return img_seam_overlay

# Compute the energy matrix from the input image
def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute X derivative of the image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Compute Y derivative of the image
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

# Compute the energy matrix using modified algorithm
def compute_energy_matrix_modified(img, rect_roi):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute X derivative of the image
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Compute Y derivative of the image
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    energy_matrix = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

    x,y,w,h = rect_roi
    # We want the seams to pass through thos region, so make sure the energy values in this region are set to 0
    energy_matrix[y:y+h, x:x+w] = 0

    return energy_matrix

# Find vertical seam in the input image
def find_vertical_seam(img, energy):
    rows, cols = img.shape[:2]
    
    # Initialize the seam vector with 0 and distance and edge matrices.
    seam = np.zeros(img.shape[0])
    dist_to = np.full(img.shape[:2], np.inf) # Set 0 value. On the first row(=top row) of image.
    dist_to[0, :] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])

    # Dynamic Programming; iterate using nested loop and compute the paths efficiently
    for row in range(rows - 1):
        for col in range(1, cols):
            # if col != 0:
            if dist_to[row+1, col-1] > dist_to[row, col] + energy[row+1, col-1]:
                dist_to[row+1, col-1] = dist_to[row, col] + energy[row+1, col-1]
                edge_to[row+1, col-1] = 1
            if dist_to[row+1, col] > dist_to[row, col] + energy[row+1, col]:
                dist_to[row+1, col] = dist_to[row, col] + energy[row+1, col]
                edge_to[row+1, col] = 0
            if col != cols - 1:
                if dist_to[row+1, col+1] > dist_to[row, col] + energy[row+1, col+1]:
                    dist_to[row+1, col+1] = dist_to[row+1, col+1] + energy[row+1, col+1]
                    edge_to[row+1, col+1] = -1

    # Retracing the path
    seam[rows-1] = np.argmin(dist_to[rows-1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i-1] = seam[i] + edge_to[i, int(seam[i])]
    
    return seam

# Remove the input vertical seam from the image
def remove_vertical_seam(img, seam):
    rows, cols = img.shape[:2]

    # To delete a point, move every point after it one step towards the left
    for row in range(rows):
        for col in range(int(seam[row]), cols-1):
            img[row, col] = img[row, col+1]
    # Discard the last colunm to create the final ouput image
    img = img[:, 0:cols-1]
    return img

# Add a vertical seam to the image
def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))

    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col-1]

        # To insert a value between two colunms, take the average value of the neighbors.
        # It looks smooth this way and we can avoid unwanted artifacts.
        for i in range(3):
            v1 = img_extended[row, int(seam[row])-1, i]
            v2 = img_extended[row, int(seam[row])+1, i]
            img_extended[row, int(seam[row]), i] = int((int(v1) + int(v2)) / 2)
    
    return img_extended

# Remove the object from the input RoI 
def remove_object(img, rect_roi):
    img_input = img.copy()
    num_seams = rect_roi[2] + 10
    print('Rectangle pts: ', rect_roi)
    energy = compute_energy_matrix_modified(img, rect_roi)

    # Start a loop and remove one seam at a time
    for i in range(num_seams):
        # Find the vertical seam that can be removed
        seam = find_vertical_seam(img, energy)

        # Remove that vertical seam
        img = remove_vertical_seam(img, seam)

        x,y,w,h = rect_roi
        # Compute energy matrix after removing the seam
        energy = compute_energy_matrix_modified(img, (x,y,w-i,h))
        print('Number of seams removed = %d' % (i+1))

    img_output = img.copy()
    cv2.imshow('Temp', img_output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Fill up the region with surrounding values so that the size of the image remains unchanged
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img = remove_vertical_seam(img, seam)
        img_output = add_vertical_seam(img_output, seam, i)
        energy = compute_energy_matrix(img)
        print('Number of seams added = %d' % (i+1))

    cv2.imshow('Input', img_input)
    cv2.imshow('Ouput Remove Object', img_output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return img_output

def reduce_image(img, num_seams):
    img_reduce = img.copy()
    energy = compute_energy_matrix(img_reduce)

    for i in range(num_seams):
        seam = find_vertical_seam(img_reduce, energy)
        # img_overlay_seam = overlay_verical_seam(img_overlay_seam, seam)
        img_reduce = remove_vertical_seam(img_reduce, seam)
        energy = compute_energy_matrix(img_reduce)
        print('Number of seams removed = %d' %(i+1))
    
    cv2.imshow('Input', img)
    cv2.imshow('Output Reduce Image', img_reduce)
    cv2.waitKey()
    cv2.destroyAllWindows()

def expand_image(img, num_seams):
    img_origin = img.copy()
    img_expand = img.copy()

    energy = compute_energy_matrix(img)
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        # img_overlay_seam = overlay_verical_seam(img_overlay_seam, seam)
        img_reduce = remove_vertical_seam(img, seam)
        img_expand = add_vertical_seam(img_expand, seam, i)
        energy = compute_energy_matrix(img)
        print('Number of seams added = %d' %(i+1))
    
    cv2.imshow('Input', img_origin)
    cv2.imshow('Output Expand Image', img_expand)
    cv2.waitKey()
    cv2.destroyAllWindows()

# if __name__ =='__main__':
#     img_input = cv2.imread(sys.argv[1])
#     num_seams = int(sys.argv[2])
#     img_reduce = img_input.copy()
#     img_expand = img_input.copy()
#     img_overlay_seam = img_input.copy()

#     # Reduce & Expand image
#     energy = compute_energy_matrix(img_reduce)
#     for i in range(num_seams):
#         seam = find_vertical_seam(img_reduce, energy)
#         img_overlay_seam = overlay_verical_seam(img_overlay_seam, seam)
#         img_reduce = remove_vertical_seam(img_reduce, seam)
#         img_expand = add_vertical_seam(img_expand, seam, i)
#         energy = compute_energy_matrix(img_reduce)
#         print('Number of seams removed = %d' %(i+1))
        

#     cv2.imshow('Input', img_input)
#     cv2.imshow('Seams', img_overlay_seam)
#     cv2.imshow('Output Reduction', img_reduce)
#     cv2.imshow('Output Extention', img_expand)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
