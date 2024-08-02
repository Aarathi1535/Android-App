import logging
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        app.logger.debug('POST request received')
        try:
            if 'file' not in request.files:
                app.logger.error('No file part in the request')
                return render_template('index.html', error="No file part in the request")

            file = request.files['file']
            if file.filename == '':
                app.logger.error('No file selected for uploading')
                return render_template('index.html', error="No file selected for uploading")

            if file:
                # Process the uploaded file here
                # For demonstration, we'll just log the filename and return a dummy response
                app.logger.debug('File %s uploaded successfully', file.filename)
                response = "Evaluate the text extracted to a score of 5"
                return render_template('index.html', response=response)

        except Exception as e:
            app.logger.error('Error occurred: %s', e)
            return render_template('index.html', error="Internal Server Error")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
