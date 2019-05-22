def get_n_rows(cols, len_set):
    return (len_set//cols + (0 if (len_set%2) == 0 else 1))

def extract_to_dict(input_path, file_depth=4, threshold=50):
    BASE, ALPHA, CHAR = range(3)
    
    zip_file = zipfile.ZipFile(input_path)
    d = {}
    for name in zip_file.namelist():
        split_path = name.split('/')

        if len(split_path) != file_depth:
            continue
            
        if split_path[-1] == '':
            continue

        if split_path[ALPHA] not in d:
            d[split_path[ALPHA]] = {}
            
        if split_path[CHAR] not in d[split_path[ALPHA]]:
            d[split_path[ALPHA]][split_path[CHAR]] = []

        f = zip_file.open(name)

        img = scipy_misc.imread(f).astype('float32')
        
        # invert image --- so high values mean 255 and low mean 0
        hi = np.where(img > threshold)
        lo = np.where(img <= threshold)
        img[lo] = 255
        img[hi] = 0
        
        d[split_path[ALPHA]][split_path[CHAR]].append(img)
        
    return d


def get_center_of_mass(image, threshold=1):
    rows, cols = np.where(image > threshold)
    return np.mean(rows), np.mean(cols)


def center_chars(char_dict, threshold=50):
    for alpha in char_dict:
        for char_id in char_dict[alpha]:
            for idx, img in enumerate(char_dict[alpha][char_id]):
                c0, r0 = img.shape
                c0 /= 2; r0 /= 2
                ri, ci = get_center_of_mass(img)
                dr = r0 - ri
                dc = c0 - ci
                
                char_dict[alpha][char_id][idx][:] = \
                    ndimage.interpolation.shift(img, [dr, dc], mode='constant', cval=0)
                
                whr = np.where(char_dict[alpha][char_id][idx] > threshold)
                char_dict[alpha][char_id][idx][whr] = 255
                
                whr = np.where(char_dict[alpha][char_id][idx] <= threshold)
                char_dict[alpha][char_id][idx][whr] = 0

                
def normalize_pixels(char_dict, max_val=1000):
    for alpha in char_dict:
        for char_id in char_dict[alpha]:
            for idx, img in enumerate(char_dict[alpha][char_id]):
                minv = img.min()
                maxv = img.max()
                
                char_dict[alpha][char_id][idx][:] = \
                    (char_dict[alpha][char_id][idx] - minv)/(maxv - minv)

#                 char_dict[alpha][char_id][idx] /= \
#                     char_dict[alpha][char_id][idx].sum()
                
#                 char_dict[alpha][char_id][idx] *= max_val

def get_min_max_sizes(char_dict):
    max_r, max_c = -np.inf, -np.inf
    min_r, min_c = np.inf, np.inf
    
    for alpha in char_dict:
        for char_id in char_dict[alpha]:
            for idx, img in enumerate(char_dict[alpha][char_id]):
                rows, cols = np.where(img > 0)
                mxr = np.max(rows)
                mxc = np.max(cols)
                mnr = np.min(rows)
                mnc = np.min(cols)
                
                max_r = max(max_r, mxr)
                max_c = max(max_c, mxc)
                
                min_r = min(min_r, mnr)
                min_c = min(min_c, mnc)

    return (max_r, max_c), (min_r, min_c)


def get_minMax_maxMin_sizes(char_dict):
    maxMin_r, maxMin_c = np.inf, np.inf
    minMax_r, minMax_c = -np.inf, -np.inf
    
    for alpha in char_dict:
        for char_id in char_dict[alpha]:
            for idx, img in enumerate(char_dict[alpha][char_id]):
                rows, cols = np.where(img > 0)
                mxr = np.max(rows)
                mxc = np.max(cols)
                mnr = np.min(rows)
                mnc = np.min(cols)
                
                maxMin_r = min(maxMin_r, mxr)
                maxMin_c = min(maxMin_c, mxc)
                
                minMax_r = max(minMax_r, mnr)
                minMax_c = max(minMax_c, mnc)

    return (maxMin_r, maxMin_c), (minMax_r, minMax_c)

    
    
def clip_images(char_dict, max_size=64):
    
    (max_r, max_c), (min_r, min_c) = get_min_max_sizes(char_dict)
    
    diff_r = max_r - min_r
    diff_c = max_c - min_c
    
    img_size = max(max(diff_r, diff_c), max_size)
    img_size += (2 if max_size == img_size else 0)
    img_size += (1 if (img_size%2) == 1 else 0)
    
    pad = (img.shape[0] - img_size)//2
    end = img_size + pad
    for alpha in char_dict:
        for char_id in char_dict[alpha]:
            for idx, img in enumerate(char_dict[alpha][char_id]):
                char_dict[alpha][char_id][idx] = img[pad:end, pad:end]
                
    
                
                
