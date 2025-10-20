from flask import Flask, render_template, request
from PIL import Image
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Fonction pour obtenir les coordonnées du rectangle englobant
def get_bounding_rect(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h)

def sort_contours(contours):
    bounding_boxes = [get_bounding_rect(contour) for contour in contours]
    tolerance = 30  # Tolérance en pixels pour considérer que deux mots sont sur la même ligne
    lines = {}
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        line_key = y // tolerance
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append((x, y, w, h, i))
    sorted_lines = sorted(lines.items(), key=lambda item: item[0])
    contours_sorted = []
    for line_key, words in sorted_lines:
        words_sorted = sorted(words, key=lambda word: word[0])
        for word in words_sorted:
            contours_sorted.append(contours[word[4]])
    return contours_sorted, sorted_lines

def clean_image(image, target_size=(40, 40)):
    """
    Cleans and preprocesses an image to reduce noise and adjust size without distortion.
    The image is centered on a square canvas and padded with white color, then resized to the target size.
    It is then converted to grayscale, blurred, and binary thresholding is applied using Otsu's method.
    """
    # Calculate the padding sizes
    height, width = image.shape[:2]
    padding_side = max(height, width)

    # Calculate padding offsets
    x_offset = (padding_side - width) // 2
    y_offset = (padding_side - height) // 2

    # Create a new square canvas to hold the padded image
    square_image = np.full((padding_side, padding_side, 3), 255, dtype=np.uint8)  # Fill the canvas with white color
    square_image[y_offset:y_offset + height, x_offset:x_offset + width] = image  # Place the original image centered

    # Resize the square image to the target size
    img_resized = cv2.resize(square_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Otsu's Thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_bounding_rect(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, w, h)

def sort_contours(contours):
    bounding_boxes = [get_bounding_rect(c) for c in contours]
    tolerance = 30
    lines = {}
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        line_key = y // tolerance
        if line_key not in lines:
            lines[line_key] = []
        lines[line_key].append((x, y, w, h, i))
    sorted_lines = sorted(lines.items(), key=lambda item: item[0])
    contours_sorted = []
    for line_key, objects in sorted_lines:
        objects_sorted = sorted(objects, key=lambda obj: obj[0])
        for obj in objects_sorted:
            contours_sorted.append(contours[obj[4]])
    return contours_sorted, sorted_lines

def load_and_predict_images(base_dir, image_size=(40, 40)):
    model = load_model('cnn_model40.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    predictions_text = ""
    object_counter = 1


    # Walk through the directory structure
    for root, dirs, files in sorted(os.walk(base_dir)):
        dirs.sort()  # Ensure directories are processed in a sorted order
        files.sort()  # Sort files to ensure they are processed in numerical order
        for file in files:
            if file.lower().endswith('.jpg'):
                path = os.path.join(root, file)
                try:
                    image = cv2.imread(path)  # Using cv2 to read the image
                    cleaned_image = clean_image(image, target_size=image_size)

                    # Model expects a batch of images
                    cleaned_image = np.expand_dims(cleaned_image, axis=0)
                    cleaned_image = np.expand_dims(cleaned_image, axis=-1)  # Add channel dimension for grayscale

                    prediction = model.predict(cleaned_image)
                    predicted_label_index = np.argmax(prediction)
                    
                    predicted_class = index_to_class[predicted_label_index]  # Adjust if necessary
                    predictions_text += predicted_class 
                    print(f"Predicted label for object {object_counter}: {predicted_class}")
                    object_counter += 1

                except Exception as e:
                    print(f"Prediction failed for object {object_counter} due to: {e}")

        predictions_text += "\n"

    return predictions_text

def predmod(image_path):
    image = cv2.imread(image_path)

    class_indices = {
        'ⴰ': 1, 'ⴱ': 2, 'ⵛ': 3, 'ⴷ': 4, 'ⴹ': 5, 'ⴻ': 6, 'ⴼ': 7, 'ⴳ': 8, 'ⵀ': 9, 'ⵉ': 10,
        'ⵊ': 11, 'ⴽ': 12, 'ⵍ': 13, 'ⵎ': 14, 'ⵏ': 15, 'ⵄ': 16, 'ⵃ': 17, 'ⵇ': 18, 'ⵔ': 19,
        'ⵕ': 20, 'ⵙ': 21, 'ⵚ': 22, 'ⵜ': 23, 'ⵟ': 24, 'ⵓ': 25, 'ⵖ': 26, 'ⵡ': 27, 'ⵅ': 28,
        'ⵢ': 29, 'ⵣ': 30, 'ⵥ': 31
    }

    # Reverse the dictionary to map from indices to class names
    index_to_class = {v: k for k, v in class_indices.items()}

    # Vérifier si l'image a été chargée correctement
    if image is None:
        print("Erreur : Impossible de charger l'image. Vérifiez le chemin.")
        exit()

    # Créer une copie de l'image originale pour l'extraction (sans modification)
    original_image = image.copy()

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage pour binariser l'image (noir/blanc inversé)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Appliquer une morphologie pour séparer les mots
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Trouver les contours de tous les mots
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_sorted, sorted_lines = sort_contours(contours)

    output_folder = "extracted_words"
    os.makedirs(output_folder, exist_ok=True)

        # Extraire et sauvegarder chaque mot détecté dans l'ordre
    object_counter = 1
    for line_key, words in sorted_lines:
        words_sorted = sorted(words, key=lambda word: word[0])
        for word in words_sorted:
            x, y, w, h, _ = word
            extracted_word = original_image[y:y+h, x:x+w]
            word_path = os.path.join(output_folder, f"word_{object_counter}.jpg")
            cv2.imwrite(word_path, extracted_word)
            object_counter += 1
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    input_folder = "extracted_words"

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, file_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erreur : Impossible de charger l'image {file_name}.")
                continue
            original_image = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_sorted, sorted_lines = sort_contours(contours)
            base_name = os.path.splitext(file_name)[0]
            image_output_folder = os.path.join(input_folder, base_name)
            os.makedirs(image_output_folder, exist_ok=True)
            object_counter = 1
            for line_key, objects in sorted_lines:
                objects_sorted = sorted(objects, key=lambda obj: obj[0])
                for obj in objects_sorted:
                    x, y, w, h, _ = obj
                    extracted_object = original_image[:, x:x+w]
                    # Calculate padding
                    padding = int(0.06 * max(w, h))
                    padded_object = cv2.copyMakeBorder(extracted_object, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    object_path = os.path.join(image_output_folder, f"object_{object_counter}.jpg")
                    cv2.imwrite(object_path, padded_object)
                    object_counter += 1
                    #cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            annotated_image_path = os.path.join(image_output_folder, "image_with_boxes.jpg")
            #cv2.imwrite(annotated_image_path, original_image)
    
    base_directory = 'extracted_words'
    predictions = load_and_predict_images(base_directory)

    return predictions






app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', prediction=None)

    imagefile = request.files['imagefile']
    
    if imagefile.filename == '':
        return render_template('index.html', prediction=None)
    
    # Ensure the static directory exists
    image_dir = "static"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Save the image temporarily
    image_path = os.path.join(image_dir, imagefile.filename)
    imagefile.save(image_path)

    # Perform OCR on the uploaded image
    try:
        # Perform OCR to extract text from the image
        ocr_text = predmod(image_path)
        
        if ocr_text.strip() == "":
            ocr_text = None  # If no text is detected, set it to None
        
        return render_template('index.html', prediction=ocr_text)
    except Exception as e:
        return render_template('index.html', prediction=None)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(port=5000, debug=True)
