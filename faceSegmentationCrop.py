import cv2
import dlib
import numpy as np
import os
from imutils import face_utils


predictor_path = "shape_predictor_68_face_landmarks.dat"
input_folder = "" #Folder path here
output_folder = os.path.join(input_folder, "Cropped")
os.makedirs(output_folder, exist_ok=True)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for image_name in os.listdir(input_folder):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        if len(rects) == 0:
            print(f"No face detected in {image_name}.")
            continue

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            jaw = shape[0:17]
            left_eyebrow = shape[17:22]
            right_eyebrow = shape[22:27]
            nose = shape[27:36]
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            mouth = shape[48:68]

            points = np.concatenate([jaw, left_eyebrow, right_eyebrow, nose, left_eye, right_eye, mouth])
            mask = np.zeros_like(gray)
            hull = cv2.convexHull(points)
            cv2.drawContours(mask, [hull], -1, (255), -1)
            
            (x, y, w, h) = cv2.boundingRect(hull)

            forehead_height = int(0.2 * h)
            y_start = max(0, y - forehead_height)
            
            y_end = min(y + h, image.shape[0])
            x_end = min(x + w, image.shape[1])

            if y_start >= y_end or x >= x_end:
                print(f"Cropping region is invalid for {image_name}.")
                continue
            
            mask_expanded = np.zeros_like(gray)
            hull_expanded = np.concatenate([shape, np.array([[x, y_start], [x + w, y_start]])])
            hull_expanded = cv2.convexHull(hull_expanded)
            cv2.drawContours(mask_expanded, [hull_expanded], -1, (255), -1)
            
            face_only = cv2.bitwise_and(image, image, mask=mask_expanded)
            face_only_cropped = face_only[y_start:y_end, x:x_end]
            
            if face_only_cropped.size == 0:
                print(f"Cropped face image is empty for {image_name}.")
                continue
            
            background = np.full_like(face_only_cropped, (255, 255, 255))
            
            mask_cropped = mask_expanded[y_start:y_end, x:x_end]
            face_only_cropped[mask_cropped == 0] = background[mask_cropped == 0]
            
            cropped_image_name = f"cropped_{i}_{image_name}"
            cropped_image_path = os.path.join(output_folder, cropped_image_name)
            cv2.imwrite(cropped_image_path, face_only_cropped)
            

print("Processing completed.")
