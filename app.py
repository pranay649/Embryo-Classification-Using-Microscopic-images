from flask import Flask, request, render_template, redirect, url_for, flash
from keras.preprocessing import image
from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input
from werkzeug.utils import secure_filename
import numpy as np
import os
import zipfile
import csv
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField
from werkzeug.utils import secure_filename
import pandas as pd

from PIL import Image


app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set the folder to store dataset-related files
DATASET_UPLOADS_FOLDER = "dataset_uploads"
app.config["DATASET_UPLOADS_FOLDER"] = DATASET_UPLOADS_FOLDER


class UploadImageForm(FlaskForm):
    folder = FileField("Image Folder", validators=[FileRequired()])
    submit = SubmitField("Upload Folder")


# Load the MobileNet model without top classification layers
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Load the trained model
model_path = "Models/Mobilenet_model.h5"  # corrected path separator
model = load_model(model_path)


# Function to predict a single Image
def predict_single_image(image_path):
    predictions = []
    probabilities_0 = []
    probabilities_1 = []

    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.convert("RGB")  # Ensure image is in RGB format
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_array)

        # Extract image ID from filename
        Image_ID = os.path.splitext(os.path.basename(image_path))[0][
            :10
        ]  # Using first 10 characters of filename as image ID

        # Predict using the model
        bottleneck_features = base_model.predict(preprocessed_img)
        prediction = model.predict(bottleneck_features)
        predicted_label = np.argmax(prediction)
        if predicted_label == 0:
            prediction_label = "Bad Embryo"
        else:
            prediction_label = "Good Embryo"
        predictions.append(prediction_label)
        probabilities = prediction.squeeze() * 100
        probabilities_0.append(probabilities[0])  # Probability for class '0'
        probabilities_1.append(probabilities[1])  # Probability for class '1'
        return predicted_label, probabilities_0, probabilities_1, Image_ID

    except Exception as e:
        print("Error:", e)
        return None, None, None


@app.route("/")
def home():
    return render_template("app.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Process the uploaded image and get the prediction
        if "image" not in request.files:
            return render_template("index.html", prediction_text="No file part")
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", prediction_text="No selected file")
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            (
                predicted_label,
                probabilities_0,
                probabilities_1,
                Image_ID,
            ) = predict_single_image(file_path)
            if predicted_label == 1:
                result = "Good Embryo"
                prediction_percentage = probabilities_1[0]
            else:
                result = "Bad Embryo"
                prediction_percentage = probabilities_0[0]
            return render_template(
                "index.html",
                result=result,
                file=file_path,
                prediction_percentage=prediction_percentage,
                Image_ID=Image_ID,
            )

    # If it's a GET request or no file is uploaded yet, just render the template
    return render_template("index.html")


test_dir = "dataset_uploads"


import os


# Function to predict labels for images in a given directory
def predict_images_in_folder(folder_path):
    predictions = []
    probabilities_0 = []
    probabilities_1 = []
    image_ids = []

    for root, dirs, files in os.walk(folder_path):
        for image_file in files:
            if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, image_file)
                img = image.load_img(image_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                preprocessed_img = preprocess_input(img_array)
                bottleneck_features = base_model.predict(preprocessed_img)
                prediction = model.predict(bottleneck_features)

                predicted_label = np.argmax(prediction)
                predictions.append(predicted_label)
                probabilities = prediction.squeeze() * 100  # Convert to percentage
                probabilities_0.append(probabilities[0])  # Probability for class '0'
                probabilities_1.append(probabilities[1])  # Probability for class '1'
                image_ids.append(os.path.splitext(image_file)[0])  # Extract image ID

    return image_ids, predictions, probabilities_0, probabilities_1


# Predict labels, probabilities, and image IDs for images in each subfolder
image_ids, predictions, probabilities_0, probabilities_1 = predict_images_in_folder(
    test_dir
)

# Create a DataFrame to store the results
results_df = pd.DataFrame(
    {
        "Image_ID": image_ids,
        "Prediction_Label": predictions,
        "Probability_Class_0": probabilities_0,
        "Probability_Class_1": probabilities_1,
    }
)

# Include only the samples where the predicted label matches the true label
correct_predictions_df = results_df

# Save the results to a CSV file
results_csv_path = "Embryo_dataset/test_results.csv"


@app.route("/generate-csv", methods=["POST"])
def generate_csv():
    # Predict labels, probabilities, and image IDs for images in each subfolder
    image_ids, predictions, probabilities_0, probabilities_1 = predict_images_in_folder(
        test_dir
    )

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "Image_ID": image_ids,
            "Prediction_Label": predictions,
            "Probability_Class_0": probabilities_0,
            "Probability_Class_1": probabilities_1,
        }
    )

    # Include only the samples where the predicted label matches the true label
    correct_predictions_df = results_df

    # Save the results to a CSV file
    results_csv_path = "Embryo_dataset/test_results.csv"
    correct_predictions_df.to_csv(results_csv_path, index=False)

    flash("CSV file generated successfully!", "success")
    return redirect(url_for("csv_page"))


@app.route("/csv", methods=["GET"])
def csv_page():
    # Read the generated CSV file
    csv_data = pd.read_csv(results_csv_path)

    # Convert CSV data to HTML table format
    csv_table = csv_data.to_html(classes="table table-striped")

    # Pass the HTML table to the template
    return render_template("csv.html", csv_table=csv_table)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    form = UploadImageForm()

    if form.validate_on_submit():
        folder = request.files["folder"]
        folder_path = os.path.join(
            app.config["DATASET_UPLOADS_FOLDER"], secure_filename(folder.filename)
        )
        folder.save(folder_path)

        # Extract the uploaded ZIP file
        try:
            with zipfile.ZipFile(folder_path, "r") as zip_ref:
                zip_ref.extractall(app.config["DATASET_UPLOADS_FOLDER"])

            # Remove the uploaded ZIP file after extraction
            os.remove(folder_path)

            # Process the extracted folder and get prediction results
            results_csv_path = predict_images_in_folder(
                app.config["DATASET_UPLOADS_FOLDER"]
            )

            # Read the prediction result CSV
            result_data = None
            if os.path.exists(results_csv_path):
                result_data = pd.read_csv(results_csv_path).values.tolist()

            flash("Dataset folder uploaded and processed successfully!", "success")
            return render_template("Upload.html", form=form, result_data=result_data)

        except Exception as e:
            flash(
                "Error occurred while processing the dataset folder: {}".format(str(e)),
                "error",
            )

    return render_template("Upload.html", form=form)


@app.route("/result")
def result():
    # Example result page
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=True)
