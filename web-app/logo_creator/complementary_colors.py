from PIL import Image


# Get the sum of max and min of a, b and c
def hilo(a, b, c):
    return min(a, b, c) + max(a, b, c)


# Get the complemented image
def get_complemented_image(r, g, b, a):
    k = hilo(r, g, b)

    r = k - r
    g = k - g
    b = k - b

    return tuple((r, g, b, a))


# Complement the image and save it
def complement(img_path):
    img = Image.open(img_path)
    size = img.size
    mode = img.mode

    in_data = img.getdata()

    out_img = Image.new(mode, size)
    out_img.putdata([get_complemented_image(*rgb) for rgb in in_data])
    out_img.save(img_path)
