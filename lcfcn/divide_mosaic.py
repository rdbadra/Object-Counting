import cv2
from skimage.io import imread
from os import walk
import os
import pickle
import numpy as np

IMAGE_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/original/'
CLICKS_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/images_clicks/'
POINTS_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/points/'
MOSAIC_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/mosaic/'
JUMP = 600
OVERLAP = 200

def has_points(h_start, h_end, w_start, w_end, clicks):
    for click in clicks:
        if click[1] <= h_end and click[1] >= h_start and click[0] <= w_end and click[0] >= w_start:
            return True
    return False

def generate_dots(h_start, h_end, w_start, w_end, clicks, mosaic_image):
    dots = np.zeros((mosaic_image.shape[0], mosaic_image.shape[1], 3), np.uint8)
    for click in clicks:
        if click[1] <= h_end and click[1] >= h_start and click[0] <= w_end and click[0] >= w_start:
            h_int = int(click[1] / (JUMP + OVERLAP))
            w_int = int(click[0] / (JUMP + OVERLAP))
            #dots = cv2.circle(output, (int(x * (224 / img.shape[1])), int(y * (224 / img.shape[0]))), radius=0, color=(0, 0, 255), thickness=1)
            x = (click[0] - ((JUMP) * w_int))
            y = (click[1] - ((JUMP) * h_int))
            print(f'Click: x: {click[0]} | y: {click[1]} ======= h_start: {h_start} | h_end: {h_end} | w_start: {w_start} | w_end: {w_end} | x: {x} | y: {y} | h_int: {h_int} | w_int: {w_int}')
            dots = cv2.circle(dots, (x, y), radius=1, color=(0, 0, 255), thickness=1)
    return dots


all_files = []
for (dirpath, dirnames, filenames) in walk(CLICKS_DIR):
    all_files.extend(filenames)
    break

images = [file.split('_file_clicks')[0] for file in all_files if '_file_clicks' in file]

for file in images:
    image = cv2.imread(os.path.join(IMAGE_DIR, file + '.jpg'))
    clicks = pickle.load(open(os.path.join(CLICKS_DIR, file + '_file_clicks'), 'rb'))
    if image is None:
        image = cv2.imread(os.path.join(IMAGE_DIR, file + '.JPG'))
    height = image.shape[0]
    width = image.shape[1]
    print(f'FILE: {file} | HEIGHT: {height} | WIDTH: {width}')
    for h_i, h in enumerate(range(0, height, JUMP)):
        for w_i, w in enumerate(range(0, width, JUMP)):
            if has_points(h, h+(JUMP + OVERLAP), w, w+(JUMP + OVERLAP), clicks):
                cv2.imwrite(f'{MOSAIC_DIR}{file}_{h}_{w}.png', image[h:h+(JUMP + OVERLAP), w:w+(JUMP + OVERLAP), :])
                dots = generate_dots(h, h+(JUMP + OVERLAP), w, w+(JUMP + OVERLAP), clicks, image[h:h+(JUMP + OVERLAP), w:w+(JUMP + OVERLAP), :])
                cv2.imwrite(f'{MOSAIC_DIR}{file}_{h}_{w}_dots.png', dots)
                print(f'Original: {image.shape} | Mosaic Image: {image[h:h+(JUMP + OVERLAP), w:w+(JUMP + OVERLAP), :].shape} | dots image: {dots.shape}' +
                    f' | Position: ({h}:{h+(JUMP + OVERLAP)}, {w}:{w+(JUMP + OVERLAP)}) | Name: {file}_{h}_{w}_dots.png')
                break
        break
    break
