import random
import math
import os
from pathlib import Path
import ffmpeg
import tempfile
import io
import src.models
import numpy as np
import cv2
from PIL import Image, ImageOps

currPath = str(Path(__file__).parent.absolute()) + '/'

# Load the autoencoder with the configuration file
cae_model_path = os.path.join(os.getcwd(), 'src/cae/model/model_yt_small_final.state')
if os.path.exists(cae_model_path):
    cae_encoder = src.models.CompressiveAE( os.path.join(os.getcwd(), 'src/cae/model/model_yt_small_final.state') ) 
else: print("Cannot find %s"%(cae_model_path))

def letterbox_image(image, size):
    '''
    Resize image with unchanged aspect ratio using padding.
    This function replaces "letterbox" and enforces non-rectangle static inferencing only
    '''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_image = np.ones((h, w, 3), np.uint8) * 114
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image, (nh, nw)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def dim_intensity(image, factor, seed=-1):
    gamma_vals = [2, 1.33, 1, 0.67, 0.5]
    adjusted = adjust_gamma(image, gamma=gamma_vals[int(factor)-1])
    return adjusted

"""
def dim_intensity(image, factor, seed=-1):
    
    Dims the intensity of the image by the give factor/range of factor. 
    
        |Parameters: 
            |image (numpy array): The original input image
            |factor (float or tuple): The diming factor (if float) or the dimimg factor range (if tuple)
            |seed (int or 1-d array_like): Seed for RandomState. Must be convertible to 32 bit unsigned integers.
        
        |Returns: 
            |image (numpy array): The dimed image  
    
    # check if factor is int (constant) or tuple (randomized range with uniform distribution):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    param = (-0.20 * factor) + 1.20
	
    #print("Intensity applied")
	
    if type(param) == float:
        #assert factor <= 1 and factor >= 0
        # adjust value channel:
        value = hsv_img[:, :, 2].astype('float64')*param
        hsv_img[:, :, 2] = value.astype('uint8')
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return image
    elif type(param) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = param
        assert upper <= 1 and upper >= 0
        assert lower <= 1 and lower >= 0
        assert upper >= lower
        random_factor = np.random.uniform(lower, upper)
        value = hsv_img[:, :, 2].astype('float64')*random_factor
        hsv_img[:, :, 2] = value.astype('uint8')
        image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        return image
    else:
        assert False, "factor type needs to be a float or tuple"
"""

def gaussian_noise(image, std, seed=-1):
    mean = 2
    if type(std) == float or type(std) == int:
        assert std > 0
        import matplotlib.pyplot as plt
        # only control standard dev:
        normal_matrix = np.random.normal(mean, std, size=image.shape)
        combined = image+normal_matrix
        np.clip(combined, 0, 255, out=combined)
        return combined.astype('uint8')
    elif type(std) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = std
        random_std = np.random.uniform(lower, upper)
        normal_matrix = np.random.normal(mean, random_std, size=image.shape)
        combined = image + normal_matrix
        np.clip(combined, 0, 255, out=combined)
        return combined.astype('uint8')
    else: assert False

# 1, 2, 3, 4, 5
def gaussian_blur(image, parameter): 
	# 13, 29, 57, 121, 293
    #p = (10*parameter) + np.power(3, parameter)
    # 15, 30, 45, 60, 75
    p = 14*parameter + 1
    parameter = int(p)
    #print("Gaussian Blur applied")
    image_copy = np.copy(image)
    cols = image_copy.shape[0]
    rows = image_copy.shape[1]
    output_image = np.zeros((cols,rows,3))
    output_image = cv2.GaussianBlur(image_copy, (parameter,parameter),0) # parameter is size of median kernel
    return output_image.astype('uint8')

def rain(image, intensity=1):
    drop_width=1
    drop_color=(200,200,200)
    drop_length=30
    drops=[]
    imshape = image.shape
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    if intensity=='0':          ## drizzle
        no_of_drops=area//770
        drop_length=10
    elif intensity=='1':        ## heavy
        drop_length=30
    elif intensity=='2':        ## torrential
        no_of_drops=area//500
        drop_length=60

    slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))

    image_t= image.copy()
    for rain_drop in drops:
            cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    new_image= cv2.blur(image_t,(5,5)) ## rainy view are blurry
    brightness_coefficient = 0.8 ## rainy days are usually shady 
    image_HLS = cv2.cvtColor(new_image,cv2.COLOR_RGB2HLS)
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB= cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB)

    return image_RGB

# 1, 2, 3, 4, 5
def saltAndPapper_noise(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    #image = image.copy()
    #p = (5 * prob) / 35.0 # 30.0 - 40.0
    p = prob / 10.0
    #print("Salt & Pepper applied")
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(image.shape[:2])
    image[probs < (p / 2)] = black
    image[probs > 1 - (p / 2)] = white
    return image

def flipAxis(image, mode):
    if mode > 0:
        return cv2.flip(image, 1) # Flips along vertical axis
    elif mode == 0:
        return cv2.flip(image, 0) # Flips along horizontal axis
    else: 
        return cv2.flip(image, -1) # Flips along both axes

def flipVertical(image):
    image = cv2.flip(image, 0)
    return image
    
def fisheye(image, factor=0.25):
    '''
    Transform image using fisheye projection
    |Parameters: 
        |image (numpy array): The original input image
        |center (array): [x, y] values of the center of transformation
        |factor (float): The distortion factor for fisheye effect
    |Returns: 
        |image (numpy array): The transformed image 
    '''
    new_image = np.zeros_like(image)
    width, height = image.shape[0], image.shape[1]
    w, h = float(width), float(height)
    for x in range(len(new_image)):
        for y in range(len(new_image[x])):
            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2*x - w)/w), float((2*y - h)/h)
            # get xn and yn euclidean distance from normalized center
            radius = np.sqrt(xnd**2 + ynd**2)
            # get new normalized pixel coordinates
            if 1 - factor*(radius**2) == 0:
                new_xnd, new_ynd = xnd, ynd
            else:
                new_xnd = xnd / (1 - (factor*(radius**2)))
                new_ynd = ynd / (1 - (factor*(radius**2)))
            # convert the new normalized distorted x and y back to image pixels
            new_x, new_y = int(((new_xnd + 1)*w)/2), int(((new_ynd + 1)*h)/2)
            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= new_x and new_x < width and 0 <= new_y and new_y < height:
                new_image[x][y] = image[new_x][new_y]
    return new_image

def barrel(image, factor=0.005):
    height, width, channel = image.shape
    k1, k2, p1, p2 = factor, 0, 0, 0
    dist_coeff = np.array([[k1],[k2],[p1],[p2]])
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y
    new_image = cv2.undistort(image, cam, dist_coeff)
    return new_image

def pick_img(start_dir):
    curr_dir = os.listdir(os.path.join(start_dir))
    # curr_dir.remove("LABELS")
    curr_path = start_dir
    
    while True:
        curr_file = random.choice(curr_dir)

        if os.path.isfile(os.path.join(curr_path, curr_file)):
            img = cv2.imread(os.path.join(curr_path, curr_file))
            if img is None:
                curr_dir = os.listdir(os.path.join(start_dir))
                # curr_dir.remove("LABELS")
                curr_path = start_dir
            else:
                return img
        else:
            curr_path = os.path.join(curr_path, curr_file)
            curr_dir = os.listdir(os.path.join(curr_path))

def simple_mosaic(image, dummy):
    # pick three images
    images = [pick_img('imgs') for x in range(4)]
    # images += [image]

    # find smallest image, resize others to fit
    smallest = image.shape[0] * image.shape[1]
    sm_shape = image.shape
    for i in images:
        curr_area = i.shape[0] * i.shape[1]
        if curr_area < smallest:
            smallest = curr_area
            sm_shape = i.shape

    # combine images into one big 2x2
    resized = [cv2.resize(curr_im, (sm_shape[0], sm_shape[1])) for curr_im in images]
    big_image = []
    big_image = np.concatenate((resized[0], resized[1]), axis=1)
    bottom = np.concatenate((resized[2], resized[3]), axis=1)
    big_image = np.concatenate((big_image, bottom), axis=0)
    
    # pick random bounds to make the mosaic image
    row_start = math.floor(random.random() * big_image.shape[0] / 2)
    col_start = math.floor(random.random() * big_image.shape[1] / 2)
    row_end = row_start + math.floor(big_image.shape[0] / 2)
    col_end = col_start + math.floor(big_image.shape[1] / 2)
    final_im = big_image[row_start:row_end][col_start:col_end]
    return final_im

def black_white(image, channel):
    channel = int(channel)
    image[:,:,0] = image[:,:,channel]
    image[:,:,1] = image[:,:,channel]
    image[:,:,2] = image[:,:,channel]
    return image 

def speckle_noise(image, std, seed=-1):
    '''
    Speckle is a granular noise that inherently exists in an image and degrades its quality. 
    It can be generated by multiplying random pixel values with different pixels of an image.
    '''
    mean = 2
    if type(std) == float or type(std) == int:
        assert std > 0
        # only control standard dev:
        gauss = np.random.normal(mean, std, size=image.shape)
    elif type(std) == tuple:
        if seed != -1:
            np.random.seed(seed)
        lower, upper = std
        random_std = np.random.uniform(lower, upper)
        gauss = np.random.normal(mean, random_std, size=image.shape)

    #gauss = gauss.reshape(image.shape[0],image.shape[1],image.shape[2]).astype('uint8')
    noise = image + image * gauss
    
    np.clip(noise, 0, 255, out=noise)
    return noise.astype('uint8')

def saturation (image, factor=50):
    '''
    Saturation impacts the color intensity of the image, making it more vivid or muted depending
    on the value.
    '''
    
    hsvimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")
    
    (h, s, v) = cv2.split(hsvimg)
    fac = 6.5025
    
    s[:] = s * ((factor/100) * fac)

    s = np.clip(s,0,255)
    imghsv = cv2.merge([h,s,v])
    
    img_sated = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return img_sated

alt_mos_dict = {}

def alternate_mosaic(image, num_slices):
    if num_slices == 1: return image
    if len(alt_mos_dict) == 0:
        # keep a copy of the last image so we know if this is
        # on a new one
        alt_mos_dict['last_img'] = image.copy()
        for i in range(2,100): alt_mos_dict[i] = []
    else:
        # if not the same image, clear out dictionary and save this one
        if (image.shape != alt_mos_dict['last_img'].shape) or np.not_equal(image, alt_mos_dict['last_img']).any():
            for i in range(2,100): alt_mos_dict[i] = []
            alt_mos_dict['last_img'] = image.copy()
    dict_ref = alt_mos_dict.get(num_slices)
    if len(dict_ref) != 0: return alt_mos_dict[num_slices]
    width, height = image.shape[0], image.shape[1]
    new_image = np.zeros_like(image)
    
    x_size = int(width/num_slices)
    while(width % x_size != 0):
        width -= 1
    x_size = int(width/num_slices)

    y_size = int(height/num_slices)
    while(height % y_size != 0):
        height -= 1
    y_size = int(height/num_slices)
    
    mats = []
    x,y = 0,0
    while x < width:
        y = 0
        while y < height:
            app = image[x:x+x_size,y:y+y_size,:]
            if len(mats) != 0:
                if app.shape != mats[0].shape: break
            mats.append(app)
            y += y_size
        x += x_size
    random.shuffle(mats)
    x,y = 0,0
    i = 0
    while x < width:
        y = 0
        while y < height:
            if i == len(mats):break
            new_image[x:x+x_size,y:y+y_size,:] = mats[i]
            y += y_size
            i += 1
        x += x_size
    alt_mos_dict[num_slices] = new_image
    return new_image

def webp_transform(image, quality=10, return_encoded=False):
    
    """
    Encodes the image using Webp image compression. 
    
        |Parameters: 
            |   image (numpy array): The original input image
            |quality (float): The quality factor for the image
        
        |Returns: 
            |image (numpy array): The dimed image  
    """
    
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    result, enc_img = cv2.imencode('.webp', image, encode_param)
    
    if return_encoded:
        return enc_img

    if result is True:
        dec_img = cv2.imdecode(enc_img, 1)
        return dec_img

def bilinear(image, percent, return_encoded=False):

    """ 
    Uses bilinear interpolation to resize the image.

    Args:
        image: The original input image
        percent: The percentage of the original image to be resized

    Returns:
        numpy array: The resized image
    """

    orig_width = image.shape[1]
    orig_height = image.shape[0]
    
    new_width = int(orig_width * (percent / 100))
    new_height = int(orig_height * (percent / 100))

    new_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    if return_encoded: return new_img

    final_img = cv2.resize(new_img, (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)

    return final_img

# 1, 2, 3, 4, 5
def ffmpeg_h264_to_tmp_video(i0, quant_lvl):
    #i0 = cv2.imread(input_file)
    param = quant_lvl #10*(int(quant_lvl) + 2)
    h,w,c = i0.shape
    # encode into png (lossless):
    i1 = cv2.imencode('.png', i0)[1]
    io_buf = io.BytesIO(i1)
    img_bytes = io_buf.getbuffer().tobytes()
    if os.name == 'nt':
        i0_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
    else:
        i0_fp = tempfile.NamedTemporaryFile(delete=True, suffix=".png", mode='wb')
    i0_fp.write(img_bytes)

    if os.name == 'nt':
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
    else:
        fp = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4", mode='wb')
    output_file = fp.name
    stream = ffmpeg.input(i0_fp.name)
    stream = ffmpeg.output(stream, output_file, vcodec="libx264", qp=param, vf="format=yuv444p,scale=%i:%i"%(w,h))
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    cap = cv2.VideoCapture(output_file)
    # assuming one frame:
    while(True):
        ret, _frame = cap.read()
        if not ret: break
        else: frame = _frame
    cap.release()
    
    i0_fp.close()
    fp.close()

    if os.name == 'nt':
        os.unlink(i0_fp.name)
        os.unlink(fp.name)

    return frame

def ffmpeg_h265_to_tmp_video(i0, quant_lvl):
    #i0 = cv2.imread(input_file)
    h,w,c = i0.shape
    # encode into png (lossless):
    i1 = cv2.imencode('.png', i0)[1]
    io_buf = io.BytesIO(i1)
    img_bytes = io_buf.getbuffer().tobytes()
    if os.name == 'nt':
        i0_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".png", mode='wb')
    else:
        i0_fp = tempfile.NamedTemporaryFile(delete=True, suffix=".png", mode='wb')

    i0_fp.write(img_bytes)

    if os.name == 'nt':
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode='wb')
    else:
        fp = tempfile.NamedTemporaryFile(delete=True, suffix=".mp4", mode='wb')

    output_file = fp.name
    stream = ffmpeg.input(i0_fp.name)
    if quant_lvl+3 > 51: quant_lvl = 51
    else: quant_lvl += 3
    stream = ffmpeg.output(stream, output_file, vcodec="libx265", pix_fmt="yuv444p", **{'x265-params':"qp=%i"%(quant_lvl)}) # the plus 3 is because of some weird ffmpeg offset thing... check the avg QP field of the ffmpeg output if u dont believe me
    ffmpeg.run(stream, overwrite_output=True, quiet=False)

    cap = cv2.VideoCapture(output_file)
    # assuming one frame:
    while(True):
        ret, _frame = cap.read()
        if not ret: break
        else: frame = _frame
    cap.release()
        
    i0_fp.close()
    fp.close()

    if os.name == 'nt':
        os.unlink(i0_fp.name)
        os.unlink(fp.name)

    return frame

# 1, 2, 3, 4, 5
def sharpen(image, param):
    #p = int(param) + 6
    #kernel = np.array([[-1,-1,-1],[-1,p,-1],[-1,-1,-1]])
    #sharp = cv2.filter2D(image, -1, kernel)
    #return sharp
    #print("Contrast applied")
    
    #sharp = cv2.addWeighted(image, (param*1.1), image, 0, (-25*param))
    
    contrast = int(-22 * param)
    alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    gamma = 127 * (1 - alpha)
	
    sharp = cv2.addWeighted(image, alpha, image, 0, gamma)
	
    return sharp
    
def rotation(image, param):
    center = tuple(np.array(image.shape[1::-1])/2)
    mat = cv2.getRotationMatrix2D(center, param, 1.0)
    rotate = cv2.warpAffine(image, mat, image.shape[1::-1])
    return rotate
    
def invert(image, param):
    invert = cv2.bitwise_not(image)
    return invert

def pincushion(image, param=0.005):
    height, width, channel = image.shape
    k1, k2, p1, p2 = 0, 0, param, param # Need to find right order of parameters for pinching
    dist_coeff = np.array([[k1],[k2],[p1],[p2]])
    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)
    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y
    new_image = cv2.undistort(image, cam, dist_coeff)
    return new_image
	
# Scale by 2^P, P = param (0 = Full, 1 = Half, 2 = Quarter, etc)
# 1, 2, 3, 4, 5
def sizescale(image, param):
    if (param == 1):
        return image
    #print("Size applied")
    scale = 1 / np.power(1.5, (param-1)) # 67% dimensional reduction
    height, width, channel = image.shape
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    rh, rw, rc = resized.shape
    background = np.zeros([height, width,3],dtype=np.uint8)
    background.fill(255)
    xoff = int(((width - rw) / 2))
    yoff = int(((height - rh) / 2))
    background[yoff:yoff+rh, xoff:xoff+rw] = resized
	
    #print(height, width)
    #print("=====")
    #print(rh, rw)
	
    return background

def cae(image, patches):  
    # Run the autoencoder with the given image
    image = cae_encoder.run(image, patches)
    return image

def jpg_compression(image, param, return_encoded=False):
    quality = 21 - (3*param)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, enc_img = cv2.imencode('.jpg', image, encode_param)
    #print("JPG Compression applied")
    if return_encoded:
        return enc_img

    if result is True:
        dec_img = cv2.imdecode(enc_img, 1)
        return dec_img


def passthrough(images, param):
    return images

#################### EXCEPTION HANDLERS ##################
def __sizescaleCheck__(param): return param > 0
def __intensityCheck__(param): return param > 0
def __gaussianNoiseCheck__(param): return param > 0
def __gaussianBlurCheck__(param): return param > 0
def __rainCheck__(param): return param >= 0 and param <= 2
def __saltPepperCheck__(param): return param > 0
#def __flipAxisCheck__(param): return True
# Changing axis constraints to allow horizontal and diagonal flip
def __flipAxisCheck__(param): return param >= -1 and param <= 1
def __fishEyeCheck__(param): return param > 0 and param <= 1
def __barrelCheck__(param): return param > 0 and param <= 0.01
def __simpleMosaicCheck__(param): return True
def __blackWhiteCheck__(param): return param in [0,1,2]
def __speckleNoiseCheck__(param): return param >= 1 and param <= 2
def __saturationCheck__(param): return param >= 0
def __altMosaicCheck__(param): return param > 0
def __bilinearCheck__(param): return param >= 0 and param <= 100
def __sharpenCheck__(param): return param > 0
#def __rotationCheck__(param): return param >= 0 and param <= 350
#def __invertCheck__(param): return True
#def __pincushionCheck__(param): return param > 0 and param <= 0.01
def __h264Check__(param): return param >= 0 and param <= 100
def __h265Check__(param): return param >= 0 and param <= 100
def __JPEGCheck__(param): return param >= 0 and param <= 100
def __WEBPCheck__(param): return param >= 0 and param <= 100
def __compressiveAutoCheck__(param): return param >= 0
def __jpgCompressionCheck__(param): return param > 0