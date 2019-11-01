import face_alignment

import cv2
import os
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt


if __name__ == '__main__':
    persons = os.listdir('movie')

    for person in persons:
        print(f'processing: {person}')
        dir_bm_eyes = f"./binary_masks/{person}"
        dir_face = f"faces/{person}/raw_faces"
        fns_face = glob(f"{dir_face}/*.*")

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        # !mkdir -p binary_masks/faceA_eyes
        Path(dir_bm_eyes).mkdir(parents=True, exist_ok=True)

        fns_face_not_detected = []
        save_path = dir_bm_eyes

        # create binary mask for each training image
        for fn in fns_face:
            raw_fn = PurePath(fn).parts[-1]
            x = plt.imread(fn)
            x = cv2.resize(x, (256, 256))
            preds = fa.get_landmarks(x)
            if preds is not None:
                preds = preds[0]
                mask = np.zeros_like(x)
                # Draw right eye binary mask
                pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
                hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                # Draw left eye binary mask
                pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
                hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
                mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                # Draw mouth binary mask
                # pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
                # hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
                # mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)
                mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
                mask = cv2.GaussianBlur(mask, (7, 7), 0)
            else:
                mask = np.zeros_like(x)
                print(f"No faces were detected in image '{fn}''")
                fns_face_not_detected.append(fn)
            plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")

        num_face = len(fns_face)

        print(f"Number of processed images: {num_face}")
        print("Number of image(s) with no face detected: " + str(len(fns_face_not_detected)))

        rand_idx = np.random.randint(num_face)
        rand_fn = fns_face[rand_idx]
        raw_fn = PurePath(rand_fn).parts[-1]
        mask_fn = f"{dir_bm_eyes}/{raw_fn}"
        im = plt.imread(rand_fn)
        mask = plt.imread(mask_fn)

        if rand_fn in fns_face_not_detected:
            print("========== No faces were detected in this image! ==========")

        fig = plt.figure(figsize=(15,6))
        plt.subplot(1,3,1)
        plt.grid('off')
        plt.imshow(im)
        plt.subplot(1,3,2)
        plt.grid('off')
        plt.imshow(mask)
        plt.subplot(1,3,3)
        plt.grid('off')
        plt.imshow((mask/255*im).astype(np.uint8))

        num_no_face_img = len(fns_face_not_detected)
        rand_idx = np.random.randint(num_no_face_img)
        x = plt.imread(fns_face_not_detected[rand_idx])
        #x = cv2.resize(x, (256,256))

        plt.grid('off')
        plt.imshow(x)
