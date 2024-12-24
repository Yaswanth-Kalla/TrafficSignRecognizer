import base64
from io import BytesIO
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
from PIL import Image

app = Flask(__name__)

# Load model pickle file
model_path = 'model1.pkl'  # Path to your model pickle file
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Classes dictionary
classes = { 1:'Speed limit (20km/h)',
           2:'Speed limit (30km/h)',
           3:'Speed limit (50km/h)',
           4:'Speed limit (60km/h)',
           5:'Speed limit (70km/h)',
           6:'Speed limit (80km/h)',
           7:'End of speed limit (80km/h)',
           8:'Speed limit (100km/h)',
           9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicle with a weight greater than 3.5 tons' }

def allowed_file(filename):
    """Check if the file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(image_stream):
    """Preprocess the image for the model."""
    img = Image.open(image_stream)
    img = img.resize((64, 64))  # Resize to match the model's expected input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and prediction."""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        # Process the image directly without saving
        img_array = preprocess_image(file.stream)
        
        # Predict the class
        prediction = model.predict(img_array)  # Adjust if your model uses predict_proba
        predicted_class_index = np.argmax(prediction)  # Get the class index
        predicted_class_index = predicted_class_index + 1
        predicted_class_name = classes.get(predicted_class_index, "Unknown")  # Get class name
        
        # Convert image to base64 to display in the result page
        img = Image.open(file.stream)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return render_template(
            'result.html',
            filename=file.filename,
            prediction=predicted_class_name,
            image_data=img_data  # Pass base64 image data to the template
        )

    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files (not needed here since we aren't saving the files)."""
    return redirect(url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    app.run(debug=True)
