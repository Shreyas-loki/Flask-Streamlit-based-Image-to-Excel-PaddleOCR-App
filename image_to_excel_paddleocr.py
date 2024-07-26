# The below commented lines should be downloaded as a prerequisite for OCR to work

'''
!git clone https://github.com/PaddlePaddle/PaddleOCR.git

!pip install paddleocr --upgrade
!pip install paddlepaddle

%cd PaddleOCR/ppstructure

# download model
!mkdir inference
%cd inference
# Download the detection model of the ultra-lightweight table English OCR model and unzip it
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar
# Download the recognition model of the ultra-lightweight table English OCR model and unzip it
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
# Download the ultra-lightweight English table inch model and unzip it
!wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar

##New OCR Model
!wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar && tar xf en_PP-OCRv3_det_infer.tar

!pip install opencv-python "tensorflow==2.*" "premailer==3.10.0" "pdf2image==1.17.0" "poppler-utils" "layoutparser==0.3.4"

%cd /content

!mkdir pages
!mkdir int_files

'''
# imports

# from pdf2image import convert_from_path
import cv2
import layoutparser as lp
from paddleocr import PaddleOCR
import tensorflow as tf
import numpy as np
import pandas as pd


# function definitions

# return the ocr output and extracted image path of table layout
def detect_table_layout_from_image(original_image_path):
    image = cv2.imread(original_image_path)
    image = image[..., ::-1]

	# load model
    model = lp.PaddleDetectionLayoutModel(config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config",
									label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
									enforce_cpu=False)  # math kernel library
	# detect
    layout = model.detect(image)
    # print(layout)

    x_1=0
    y_1=0
    x_2=0
    y_2=0

    for l in layout:
        # print(l)
        if l.type == 'Table':
            x_1 = int(l.block.x_1)
            y_1 = int(l.block.y_1)
            x_2 = int(l.block.x_2)
            y_2 = int(l.block.y_2)
            break
    # print(x_1,y_1,x_2,y_2)

    cv2.imwrite(f'./int_files/extracted_image.jpg', image[y_1:y_2,x_1:x_2])


    ocr = PaddleOCR(lang='en')

    extracted_image_path = f'./int_files/extracted_image.jpg'
    # image_cv = cv2.imread(extracted_image_path)
    # image_height = image_cv.shape[0]
    # image_width = image_cv.shape[1]

    ocr_output = ocr.ocr(extracted_image_path)[0]
    # output contains all the text recognized boxes with confidence level
    print("Output:\n", ocr_output)


    boxes = [line[0] for line in ocr_output]  # 4 co-ordinates values for each character
    texts = [line[1][0] for line in ocr_output] # text value present inside each box
    probabilities = [line[1][1] for line in ocr_output] # confidence level of each character
    print("Texts:\n", texts)

    image_boxes = cv2.imread(extracted_image_path)

    for box,text in zip(boxes,texts):
        cv2.rectangle(image_boxes, (int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1])),(0,0,255),1)
        cv2.putText(image_boxes, text,(int(box[0][0]),int(box[0][1])),cv2.FONT_HERSHEY_SIMPLEX,1,(222,0,0),1)

    cv2.imwrite(f'./int_files/detections.jpg', image_boxes)
    

    im = cv2.imread(extracted_image_path)

    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0, int(box[0][0])
        y_h, y_v = int(box[0][1]), 0

        image_height = im.shape[0]
        image_width = im.shape[1]

        width_h, width_v = image_width, int(box[2][0] - box[0][0])
        height_h, height_v = int(box[2][1] - box[0][1]), image_height

        horiz_boxes.append([x_h, y_h, x_h+width_h, y_h+height_h])
        vert_boxes.append([x_v, y_v, x_v+width_v, y_v+height_v])

        cv2.rectangle(im, (x_h,y_h), (x_h+width_h, y_h+height_h), (0,0,255), 1)
        cv2.rectangle(im, (x_v,y_v), (x_v+width_v, y_v+height_v), (0,255,0), 1)

    cv2.imwrite(f'./int_files/horiz_vert_boxes.jpg', im)

    horiz_out = tf.image.non_max_suppression(
                            horiz_boxes,
                            probabilities,
                            max_output_size = 1000,
                            iou_threshold=0.1,
                            score_threshold=float('-inf'),
                            name=None
                )

    horiz_lines = np.sort(np.array(horiz_out))

    im_nms = cv2.imread(extracted_image_path)

    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)

    cv2.imwrite(f'./int_files/image_nm_suppression.jpg', im_nms)

    vert_out = tf.image.non_max_suppression(
                            vert_boxes,
                            probabilities,
                            max_output_size = 1000,
                            iou_threshold=0.1,
                            score_threshold=float('-inf'),
                            name=None
                )

    vert_lines = np.sort(np.array(vert_out))
    print("Number of columns:", len(vert_lines))

    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)

    cv2.imwrite(f'./int_files/image_nm_suppression.jpg',im_nms)


    # texts & vert_lines - are the two variables needed to reconstruct the table
    result = []
    columns = len(vert_lines)

    for i in range(0, len(texts), columns):
        result.append(texts[i:i + columns])

    out_array = np.array(result)
    print("Output array:\n", out_array)

    pd.DataFrame(out_array).to_csv(f'./converted_files/output_excel.csv', index=False)

    return "Converted to excel successfully!"

