import pytesseract
from pytesseract import Output
# Import OpenCV library
import cv2
from pathlib import Path


def read_image(image_path: str or Path):
   
    if not isinstance(image_path, str):
        image_path_string = str(image_path)
    
    cv2_image = cv2.imread(image_path_string)
    
    return cv2_image

def resize_image(cv2_image):
    
    height, width, channel = cv2_image.shape
    resized = cv2.resize(cv2_image, (width//2, height//2))

    return resized, (height, width)

def preprocess_image(cv2_image):

    # Convert image to grey scale
    gray_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    # Converting grey image to binary image by Thresholding
    thresh_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return thresh_img

def get_text_bbox(ocr_details):
    
    n_boxes = len(ocr_details['level'])

    text_bbox = []

    # Extract and draw rectangles for all bounding boxes
    for i in range(n_boxes):

        top_left_x, top_left_y, = ocr_details['left'][i], ocr_details['top'][i]
        w, h = ocr_details['width'][i], ocr_details['height'][i]

        bottom_right_x, bottom_right_y = top_left_x + w, top_left_y + h

        bbox = [top_left_x, top_left_y , bottom_right_x, bottom_right_y]

        text = ocr_details['text'][i]

        if text == "": continue

        text_bbox.append(
            {
                "text": text,
                "bbox": bbox
            }
        )
    
    return text_bbox


def get_ocr_word_box_list(img_path: str or Path):

    cv2_image = read_image(
        image_path = img_path
    )

    resized_image, shape = resize_image(
        cv2_image = cv2_image
    )

    preprocessed_image = preprocess_image(
        cv2_image = resized_image
    )

    # configuring parameters for tesseract
    custom_config = r'--oem 3 --psm 6'

    # Get all OCR output information from pytesseract
    ocr_output_details = pytesseract.image_to_data(
        preprocessed_image,
        output_type = Output.DICT,
        config=custom_config,
        lang='eng'
    )

    text_bbox = get_text_bbox(
        ocr_details = ocr_output_details
    )

    return text_bbox, shape

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ] 