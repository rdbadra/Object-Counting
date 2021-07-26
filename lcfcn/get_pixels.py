import urllib
import cv2
from os import walk
import numpy as np
import pickle

IMAGE_DIR = '/home/roberto/Descargas/ImagenesMicroplasticos/'

OUTPUT_FOLDER = 'images_clicks'

all_files = []
for (dirpath, dirnames, filenames) in walk(f'{IMAGE_DIR}original'):
    all_files.extend(filenames)
    break

resized_images = []
for (dirpath, dirnames, filenames) in walk(f'{IMAGE_DIR}{OUTPUT_FOLDER}'):
    resized_images.extend(filenames)
    break
print(resized_images)

pngs = [file.split('_file_clicks')[0].lower() for file in resized_images if '_file_clicks' in file]

files = []
for file in all_files:
    if(file.lower().split('.jpg')[0] not in pngs
    and file.split('.')[1] not in ['png', 'txt']):
        files.append(file)

#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 8
    
    # Blue color in BGR
    color = (0, 255, 255)
    
    # Line thickness of 2 px
    thickness = 5
    if event == 1:
        global file_clicks
        global img
        global output
        file_clicks.append([x, y])
        #file_clicks.append([int(x * (224 / img.shape[1])), int(y * (224 / img.shape[0]))])
        #store the coordinates of the right-click event
        output = cv2.circle(output, (int(x * (224 / img.shape[1])), int(y * (224 / img.shape[0]))), radius=0, color=(0, 0, 255), thickness=1)

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(file_clicks)
        img = cv2.putText(img, f'{len(file_clicks)}', (x, y), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
        img = cv2.circle(img, (x, y), radius=10, color=(0, 0, 255), thickness=20)
        cv2.imshow('image', img)

def write_file(image_name, output):
    global file_clicks
    output = cv2.resize(output, (224, 224), interpolation=cv2.INTER_AREA)
    # cv2.imwrite(f'{IMAGE_DIR}{OUTPUT_FOLDER}/{image_name.split(".")[0]}_dots.png', output)
    pickle.dump(file_clicks, open(f'{IMAGE_DIR}{OUTPUT_FOLDER}/{image_name.split(".")[0]}_file_clicks', "wb" ) )
    '''
    with open(f'{IMAGE_DIR}{image_name.split(".")[0]}.txt', 'w') as file:
        file.write(f'X;Y\n')
        for file_click in file_clicks:
            file.write(f'{file_click[0]};{file_click[1]}\n')
    '''

for image_name in files:
    print(image_name)
    file_clicks = []
    path_image = f'{IMAGE_DIR}original/' + image_name
    img = cv2.imread(path_image)
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    height = img.shape[0]
    width = img.shape[1]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    output = np.zeros((224, 224, 3), np.uint8)

    #set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    print(k)
    # Press Enter
    if(k == 13):
        write_file(image_name, output)
        '''
        img = cv2.imread(path_image)
        output = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{IMAGE_DIR}{OUTPUT_FOLDER}/{image_name.split(".")[0]}.jpg', output)
        '''
    #cv2.waitKey(0)
    # Press Space
    if (k == 32):
        cv2.destroyAllWindows()
        break