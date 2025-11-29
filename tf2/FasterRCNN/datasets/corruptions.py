import numpy as np
import PIL
import skimage as sk

def clipped_zoom(img, zoom_factor):
    h = img.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))

    top = (h - ch) // 2
    img = scizoom(img[:,top:top + ch, top:top + ch], (1,zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[1] - h) // 2

    return img[:,trim_top:trim_top + h, trim_top:trim_top + h]

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x)/255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1)*255 
    
def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]
    #std = np.std(train_images, axis=(0, 1, 2, 3))
    #mu = np.mean(train_images, axis=(0, 1, 2, 3))
    x = np.array(x) #*(std+1e-7) + mean + 1.5258789e-05
    print(x.min())
    x = np.array(x)/ 255.
    return (np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255)


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x)/255. , mode='s&p', amount=c)
    return np.clip(x, 0, 1) *255

def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x)).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, -1.8816435, 2.0934134)

def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .04][severity - 1]

    x = np.array(x) / 255.0
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
    
def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) #/ 255.
    x = sk.color.rgb2hsv(x)
    x[:,:, :, 1] = np.clip(x[:,:, :, 1] * c[0] + c[1], -1.8816435, 2.0934134)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) #* 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x