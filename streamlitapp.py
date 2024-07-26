
import streamlit as st
import os
import image_to_excel_paddleocr as ocr
from werkzeug.utils import secure_filename

# '''
def convert_image():
    try:
        result = ""
        # write the functionality to pass the image path to OCR model
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'pages', secure_filename(uploaded_file.name))

        result = ocr.detect_table_layout_from_image(file_path)

    except Exception as e:
        print("Exception:", e)
        result = "Failed to convert to Excel!"
        
    finally:
        return result


# driver code
st.set_page_config(page_title="Image to Excel Converter")
st.header("Upload the Image file to be converted to Excel")

# Get the file from file uploader
uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    result = convert_image()

    print("Converted successfully!!")
    st.subheader(result)

