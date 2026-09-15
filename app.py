
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import time

from lime import lime_image
from skimage.segmentation import mark_boundaries,slic

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "brain_tumor_model_95.keras"

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# CLASS ORDER
#
# IMPORTANT:
# This order MUST be the same order used when training
# brain_tumor_model_95.keras
# ============================================================

categories = [
    "No Tumor",
    "Meningioma",
    "Pituitary Tumor",
    "Glioma"
]


# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = 128


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    "static",
    exist_ok=True
)


# ============================================================
# LIME PREDICTION FUNCTION
# ============================================================

def predict_fn(images):

    # Convert incoming images to NumPy array
    images = np.array(images)

    # Make sure the datatype is correct
    images = images.astype(
        "float32"
    )

    # SAME NORMALIZATION AS CNN PREDICTION
    images = images / 255.0

    # Return prediction probabilities
    return model.predict(
        images,
        verbose=0
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    prediction = None
    confidence = None
    image_path = None
    lime_image_path = None


    # ========================================================
    # IMAGE UPLOAD
    # ========================================================

    if request.method == "POST":

        file = request.files.get(
            "image"
        )


        if file and file.filename:

            # =================================================
            # SAVE UPLOADED IMAGE
            # =================================================

            image_path = os.path.join(
                UPLOAD_FOLDER,
                file.filename
            )

            file.save(
                image_path
            )


            # =================================================
            # LOAD IMAGE
            # =================================================

            image = Image.open(
                image_path
            ).convert("RGB")


            # =================================================
            # RESIZE IMAGE
            #
            # This matches the 128 x 128 model input.
            # =================================================

            image = image.resize(
                (
                    IMG_SIZE,
                    IMG_SIZE
                )
            )


            # =================================================
            # CONVERT IMAGE TO NUMPY
            # =================================================

            image_array = np.array(
                image
            )


            # =================================================
            # PREPARE IMAGE FOR CNN
            # =================================================

            model_input = (
                np.expand_dims(
                    image_array,
                    axis=0
                ).astype(
                    "float32"
                ) / 255.0
            )


            # =================================================
            # CNN PREDICTION
            # =================================================

            predictions = model.predict(
                model_input,
                verbose=0
            )


            # =================================================
            # GET PREDICTED CLASS
            # =================================================

            predicted_index = int(
                np.argmax(
                    predictions[0]
                )
            )


            prediction = categories[
                predicted_index
            ]


            # =================================================
            # GET CONFIDENCE
            # =================================================

            confidence = (
                float(
                    predictions[0][
                        predicted_index
                    ]
                ) * 100
            )


            # =================================================
            # LIME EXPLAINER
            # =================================================

            explainer = (
                lime_image.LimeImageExplainer()
            )


            # =================================================
            # SLIC SEGMENTATION
            #
            # LIME divides the image into meaningful
            # superpixel regions.
            # =================================================

            def segmentation_fn(img):

                return slic(
                    img,
                    n_segments=60,
                    compactness=8,
                    sigma=1,
                    start_label=0
                )


            # =================================================
            # GENERATE LIME EXPLANATION
            # =================================================

            explanation = (
                explainer.explain_instance(

                    image_array,

                    predict_fn,

                    labels=(
                        predicted_index,
                    ),

                    hide_color=0,

                    num_samples=1500,

                    segmentation_fn=segmentation_fn
                )
            )


            # =================================================
            # GET LIME IMPORTANT REGIONS
            #
            # positive_only=False:
            #
            # Shows both regions that support the prediction
            # and regions that work against the prediction.
            #
            # num_features=15:
            #
            # Shows the 15 most important regions.
            # =================================================

            temp, mask = (
                explanation.get_image_and_mask(

                    predicted_index,

                    positive_only=False,

                    num_features=15,

                    hide_rest=False
                )
            )


            # =================================================
            # CREATE LIME BOUNDARY VISUALIZATION
            # =================================================

            lime_visualization = (
                mark_boundaries(

                    temp / 255.0,

                    mask
                )
            )


            # =================================================
            # CREATE UNIQUE FILE NAME
            # =================================================

            filename = (
                "lime_result_"
                + str(
                    int(
                        time.time()
                    )
                )
                + ".png"
            )


            lime_full_path = os.path.join(
                "static",
                filename
            )


            # =================================================
            # CREATE SIDE-BY-SIDE FIGURE
            # =================================================

            fig, ax = plt.subplots(
                1,
                2,
                figsize=(12, 5)
            )


            # =================================================
            # ORIGINAL MRI
            # =================================================

            ax[0].imshow(
                image_array
            )

            ax[0].set_title(
                "Original MRI",
                fontsize=13
            )

            ax[0].axis(
                "off"
            )


            # =================================================
            # LIME EXPLANATION
            # =================================================

            ax[1].imshow(
                lime_visualization
            )

            ax[1].set_title(

                "LIME Important Regions\n"
                f"Prediction: {prediction}\n"
                f"Confidence: {confidence:.2f}%",

                fontsize=12
            )

            ax[1].axis(
                "off"
            )


            # =================================================
            # SAVE LIME IMAGE
            # =================================================

            plt.tight_layout()

            plt.savefig(
                lime_full_path,
                bbox_inches="tight",
                dpi=150
            )

            plt.close()


            # =================================================
            # SEND FILE NAME TO HTML
            # =================================================

            lime_image_path = filename


    # ========================================================
    # RENDER HTML
    # ========================================================

    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        image_path=image_path,

        lime_image_path=lime_image_path
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

