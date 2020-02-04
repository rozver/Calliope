from PIL import Image


def hilo(a, b, c):
    return min(a, b, c) + max(a, b, c)

"""
def get_complement_image_name(img_path):
    path_split = img_path.split('/')
    img_name = 'complemented-'+path_split[-1]
    return img_name
 """


def get_complemented_image(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))


def complement(img_path):
    img = Image.open(img_path)
    size = img.size
    mode = img.mode

    in_data = img.getdata()

    out_img = Image.new(mode, size)
    out_img.putdata([get_complemented_image(*rgb) for rgb in in_data])
    out_img.save(img_path)
