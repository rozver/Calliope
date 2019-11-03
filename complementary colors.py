import sys
from PIL import Image

def hilo(a, b, c):
    if c < b: b, c = c, b
    if b < a: a, b = b, a
    if c < b: b, c = c, b
    return a + c

def complement(r, g, b):
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))

def complement_image(oname):
    print('Loading', "D:\\Снимки\\Wallpapers\\Wallpaper\\a6105f2351783c6df7e6f3efa60d588f.jpg")
    img = Image.open("D:\\Снимки\\Wallpapers\\Wallpaper\\a6105f2351783c6df7e6f3efa60d588f.jpg")
    img.show()

    size = img.size
    mode = img.mode
    in_data = img.getdata()

    print('Complementing...')
    out_img = Image.new(mode, size)
    out_img.putdata([complement(*rgb) for rgb in in_data])
    out_img.show()
    out_img.save(oname)
    print('Saved to', oname)

complement_image("D:\\Снимки\\Wallpapers\\Wallpaper\\output.png")