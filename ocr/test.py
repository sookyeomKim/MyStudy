import cv2
import numpy as np
from PIL import Image
from pytesseract import image_to_string

# Path of working folder on Disk
src_path = "images/"


def get_string(img_path):
    # 이미지를 불러온다
    img = cv2.imread(img_path)

    # 회색 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 확장 및 노이즈 제거
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # 위 작업 진행된 이미지 제작
    cv2.imwrite(src_path + "removed_noise.png", img)

    #  이미지의 흑백화를 할 수 있는데까지 진행
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # 위 작업 진행된 이미지 제작
    cv2.imwrite(src_path + "thres.png", img)

    # 이미지의 텍스트를 추출
    result = image_to_string(Image.open(src_path + "thres.png"))

    # os.remove(temp)

    return result


print('--- Start recognize text from image ---')

result = get_string(src_path + "6.png")
print(result)

print("------ Done -------")
