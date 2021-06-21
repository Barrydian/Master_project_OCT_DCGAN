import os, shutil
import cv2

def resize_img (src_database, target_database, width, height, channel) :
    
    if ( os.path.exists(src_database)) :

        if os.path.exists(target_database):    
            shutil.rmtree(target_database)
        os.mkdir(target_database)

        for each in os.listdir(src_database):
            if channel==1:
                img = cv2.imread(os.path.join(src_database,each), cv2.IMREAD_GRAYSCALE)
            elif channel==3:
                img = cv2.imread(os.path.join(src_database,each), cv2.IMREAD_COLOR)

            img = cv2.resize(img,(width,height))
            cv2.imwrite(os.path.join(target_database,each), img) 
            print(os.path.join(target_database,each))
        print(' --- Images resizing done ... ')