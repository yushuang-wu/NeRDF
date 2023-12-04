from PIL import Image
import os

down_rate = 16
img_path = "data/nerf_llff_data/fern/images/"
img_newd = f"data/nerf_llff_data/fern/images_{down_rate}/"
if not os.path.exists(img_newd):
    os.mkdir(img_newd)

imgs = os.listdir(img_path)
for i in imgs:
    im = Image.open(img_path+i)
    new_size = im.size[0]//down_rate, im.size[1]//down_rate

    im_new = im.resize(new_size)
    im_new.save(img_newd+i)