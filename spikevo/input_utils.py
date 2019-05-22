import numpy as np
import scipy as sp
from scipy import ndimage

def get_receptive_fields_indices(width, height, receptive_size, row_stride, 
    col_stride=None, boundary=None):
    """
        width: Horizontal size of image, in pixels (neurons)
        height: Vertical size of image, in pixels (neurons)
        receptive_size: Horizontal and vertical size of input window
        row_stride: Spatial sampling frequency in the horizontal axis 
                    (sample horizontally each row_stride pixels)
        col_stride: Spatial sampling frequency in the vertical axis 
                    (sample vertically each col_stride pixels). If set to
                    None it will use row_stride as its value.
        boundary: How to handle boundary conditions. None means valid input 
                  only (no fake inputs) #todo: add more options!
        
        return:  A dictionary with the indices per region centered at 
                 key0, key1. In this context key0 is the first level key, 
                 key1 is the second level key; they are equal to the row 
                 and column, respectively.
            
    """
    if col_stride is None:
        col_stride = row_stride

    half_rs = receptive_size//2
    max_row = height - half_rs
    max_col = width - half_rs
    isize = width*height
    
    all_indices = {}
    for row in range(half_rs, max_row, row_stride):
        #get the dictionary for current row or an empty one
        row_dict = all_indices.get(row, {})
        indices = []
        current_rows = np.arange(max(0, row-half_rs), min(height, row + half_rs+1))
        for col in range(half_rs, max_col, col_stride):
            current_cols = np.arange(max(0, col-half_rs), min(width, col + half_rs+1))
            
            current_indices = np.repeat(current_rows, len(current_cols)) * width + \
                              np.tile(current_cols, len(current_rows))
            
            row_dict[col] = current_indices

        all_indices[row] = row_dict
    
    return all_indices


def get_angled_bar(image_size, bar_width, angle, fill_val=1.0, threshold=None):
    """
        image_width: image_size refers to both width and height of the 
                    image (i.e. it's a square input)
        bar_width: by default the bar will be as high as the image, how 
                   wide we can set with this parameter
        angle: at which angle to rotate the bar (in degrees)
        fill_val: what value to put in the original line
        threshold: lower limit for values to be 'active' (i.e. v < threshold == 0)
        return: a b/w (image_size x image_size) image with a bar rotated 
                angle degrees.
    """
    
    dbl_s = image_size*2 - 1
    hlf_s = image_size//2
    hlf_bw = bar_width//2

    tmp = np.zeros((dbl_s, dbl_s))
    tmp[image_size-hlf_bw-1: image_size + hlf_bw, :] = fill_val
    
    tmp[:] = ndimage.rotate(tmp, angle, mode='reflect', reshape=False)
    tmp[tmp < 0] = 0
    if threshold is not None:
        tmp[tmp < threshold] = 0

    return tmp[hlf_s: hlf_s + image_size, hlf_s: hlf_s + image_size] 
