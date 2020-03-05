from PIL import Image


# Get the sum of max and min of a, b and c
def hilo(a, b, c):
    return min(a, b, c) + max(a, b, c)


# Add 'complemented-' prefix to the image name
def get_complement_image_name(img_path):
    path_split = img_path.split('/')
    img_name = 'complemented-'+path_split[-1]
    return img_name


# Complement red, green, blue values
def complement(r, g, b, a):
    k = hilo(r, g, b)

    r = k - r
    g = k - g
    b = k - b

    return tuple((r, g, b, a))


# Complement image and save it with new name
def complement_image(img_path):
    img = Image.open(img_path)
    size = img.size
    mode = img.mode

    in_data = img.getdata()
    print(in_data)

    out_img = Image.new(mode, size)
    out_img.putdata([complement(*rgb) for rgb in in_data])
    out_img.save(get_complement_image_name(img_path))


if __name__ == '__main__':
    img_path = input()
    complement_image(img_path)
