import time
import PIL.Image as Image
import torchvision.transforms as transforms
from tqdm import tqdm
import os


# Set5
photo_path = './results_quant_copy/out_v2_set5'

test_list = os.listdir(photo_path)

t = time.time()

for imgname in tqdm(test_list):
    photo_str = photo_path + '/' + imgname

    img = Image.open(photo_str).convert('RGB')
    img1 = transforms.ToTensor()(img).unsqueeze(0).cuda()
    h, w = img.size

    img_LR4 = img.resize((int(h * 4), int(w * 4)), Image.BICUBIC)

    img_LR4.save("new_path")
