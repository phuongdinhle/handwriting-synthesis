import os
import sys
import cv2
import pdb
import json
import numpy as np
from demo import Hand
from cairosvg import svg2png
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.transform import rescale

# class EmailGenerator():
#     '''Generate a plausible looking Japanese email'''

#     def __init__(self):
#         json_list = ['male', 'female', 'surnames']
#         part_dict = dict.fromkeys(json_list)
#         for part in json_list :
#             with open('names/{}.json'.format(part)) as f:
#                 names = json.load(f)
#                 part_dict[part] = names

#         part_dict['names'] = part_dict['male'] + part_dict['female']
#         part_dict.pop('male')
#         part_dict.pop('female')

#         email_domains = ['icloud.com', 'icould.com', 'yahoo.co.jp', 'gmail.com']
#         part_dict['domains'] = email_domains
#         self.part_dict = part_dict
        
#     def create(self):
#         '''Generate a random Japanese email'''
#         part_dict = self.part_dict
#         first = np.random.choice(part_dict['names']).lower()
#         last = np.random.choice(part_dict['surnames']).lower()
#         number = np.random.randint(100, 10000)
#         domain = np.random.choice(part_dict['domains'])
#         email = '{}.{}-{}(a){}'.format(first, last, number, domain)
#         return email

def generate_random_string():
    # vocab = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.')
    vocab = list('種子')

    n_chars = np.random.randint(1, 30)
    result = ''
    for i in range(n_chars):
        result += np.random.choice(vocab)
    return result

def generate_image(hand):
    try:
        string = generate_random_string()
        lines = [string]

        biases = [0.75 for i in lines]
        print(biases)
        style_list = [np.random.randint(9) for i in lines]
        print (style_list)
        styles = [np.random.choice(style_list)]
        stroke_colors = ['black']
        stroke_widths = [np.random.uniform(1.2, 3.5)]
        hand.write(
            filename='tmp.svg',
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths
        )
        img = svg2png(url='tmp.svg')
        img = np.frombuffer(img, 'uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = rgb2gray(img)
        img = img < 0.5
        img = img.astype('uint8')

        region = list(regionprops(img))[0]
        minr, minc, maxr, maxc = region.bbox
        img = img[minr:maxr, minc:maxc]
    except:
        return None, None
    return img, string

def generate_samples(n_samples, path):
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    src_path = os.path.join(path, 'src-train.txt')
    lbl_path = os.path.join(path, 'tgt-train.txt')

    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            start = len(f.read().splitlines())
    else:
        start = 0
        
    hand = Hand()
    
    for i in range(start, start + n_samples):
        img = None
        while img is None:
            img, email = generate_image(hand)

        # Random bleed-in
        bleeds = []
        for _ in [0, 1]:
            bleed = np.random.randint(-3, 3)
            if bleed < 0:
                bleed = 0
            bleeds.append(bleed)

        y_bleed_0 = bleeds[0]
        y_bleed_1 = bleeds[1]

        img = img[y_bleed_0:img.shape[0] - y_bleed_1, :]

        # Random pad widtth
        y_pad_0, y_pad_1 = np.random.randint(6), np.random.randint(6)
        x_pad_0, x_pad_1 = np.random.randint(21), np.random.randint(21)
        img = np.pad(img, ((y_pad_0, y_pad_1), (x_pad_0, x_pad_1)), 'constant')

        # Random rescale
        scaling_factor = np.random.randn()/5 + 1
        img = rescale(img, scaling_factor, preserve_range=True)

        # Convert to 3 channels
        img = img*255
        img = np.stack((img,) * 3, -1)

        filename = '{:07}.png'.format(i)
        
        if not os.path.exists(os.path.join(path, 'images')):
            os.mkdir(os.path.join(path, 'images'))
        
        img = 255 - img
        cv2.imwrite(os.path.join(path, 'images', filename), img)

        filename = filename + '\n'
        
        email = email.replace('(a)', '@')
        email = ' '.join(email) + '\n'
        
        if os.path.exists(src_path) or os.path.exists(lbl_path):
            mode = 'a'
        else:
            mode = 'w'

        with open(src_path, mode) as f:
            f.write(filename)

        with open(lbl_path, mode) as f:
            f.write(email)

        print('Generated {} samples'.format(i + 1))
        
    os.remove('tmp.svg')

if __name__ == '__main__':
    n_samples = int(sys.argv[1])
    path = sys.argv[2]
    generate_samples(n_samples, path)
