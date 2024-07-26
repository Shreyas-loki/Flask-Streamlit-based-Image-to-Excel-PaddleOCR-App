# Using flask to make an api
# import necessary libraries and functions
 
import os
from flask import Flask, jsonify, request, make_response, render_template
import image_to_excel_paddleocr as ocr
from werkzeug.utils import secure_filename


# creating a Flask app
app = Flask(__name__)


@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')
 

@app.route('/convertImage', methods=['GET', 'POST'])
def convertImage():
    try:
        if request.method == 'POST':
            # Get the file from post request
            f = request.files['file']

            # Save the file to ./pages
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'pages', secure_filename(f.filename))
            f.save(file_path)

            result = ocr.detect_table_layout_from_image(file_path)
            print("Converted successfully!!")

            return result
    
    except Exception as e:
        print("Exception in executing API", e)
        return make_response(jsonify({"Status": str(e)}), 500)

    

if __name__ == '__main__':
    app.run(debug=True)