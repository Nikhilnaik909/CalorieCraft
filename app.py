from flask import Flask, render_template, request, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
import re
import torch
import json
import urllib.parse
import urllib.request
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from werkzeug.utils import secure_filename


# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# AI FOOD RECOGNITION MODEL
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "nateraw/food"

print("Loading Food Recognition Model...")

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

model.to(device)
model.eval()

print("Food Recognition Model Loaded Successfully!")
print("Using device:", device)


# ==========================================
# LOAD FOOD DATASETS
# ==========================================

# Food recommendations continue to use food_exercise.csv.
data = pd.read_csv('./dataset/food_exercise.csv')

# Image recognition nutrition matching uses food.csv.
recognition_data = pd.read_csv('./dataset/food.csv')

numeric_columns = ['Calories', 'Proteins', 'Fats', 'Carbohydrates']

for column in numeric_columns:
    data[column] = pd.to_numeric(data[column], errors='coerce')
    recognition_data[column] = pd.to_numeric(
        recognition_data[column], errors='coerce'
    )

data['Name'] = data['Name'].fillna('').astype(str)
recognition_data['Name'] = recognition_data['Name'].fillna('').astype(str)


# ==========================================
# FOOD NAME MATCHING
# ==========================================

def normalize_food_name(name):
    """Convert food names into a consistent format."""

    name = str(name).lower().strip()
    name = re.sub(r'[_-]+', ' ', name)
    name = re.sub(r'[^a-z0-9 ]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name


# Explicit aliases for Food-101 labels.
# Each alias must correspond to a food name that actually exists
# in dataset/food.csv to return a nutrition match.
FOOD_ALIASES = {
    'chicken curry': 'Chicken Curry',
    'greek salad': 'Greek Salad',
    'caesar salad': 'Caesar Salad',
    'french fries': 'French Fries',
    'pulled pork sandwich': 'Pulled Pork Sandwich',
    'chicken biryani': 'Chicken Biryani'
}


def match_food_with_dataset(predicted_food):
    """
    Find the recognized food in the nutrition dataset.

    Returns a list of matching food dictionaries.
    Generic 'ice cream' returns both Saffron Ice Cream
    and Matcha Ice Cream when those entries exist.
    """

    normalized_name = normalize_food_name(predicted_food)

    # Special handling for generic ice cream.
    if normalized_name == "ice cream":
        ice_cream_names = {
            "saffron ice cream",
            "matcha ice cream"
        }

        matching_foods = recognition_data[
            recognition_data['Name'].apply(
                lambda name: normalize_food_name(name) in ice_cream_names
            )
        ]

        return matching_foods.to_dict(orient='records')

    # Apply an explicit alias if one is available.
    search_name = FOOD_ALIASES.get(
        normalized_name, normalized_name
    )

    matching_foods = recognition_data[
        recognition_data['Name'].apply(
            lambda name: normalize_food_name(name)
            == normalize_food_name(search_name)
        )
    ]

    if matching_foods.empty:
        return []

    # Return a list so the upload route handles one or many matches
    # in exactly the same way.
    return [matching_foods.iloc[0].to_dict()]


# ==========================================
# FOOD IMAGE RECOGNITION
# ==========================================

def recognize_food(image_path):
    """Predict a food label and return its confidence percentage."""

    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=-1
        )

        food_name = model.config.id2label[predicted_index.item()]
        confidence_score = confidence.item() * 100

        print("Predicted food:", food_name)
        print("Confidence:", round(confidence_score, 2), "%")

        # Reject predictions below the current confidence threshold.
        if confidence_score < 60:
            return None, confidence_score

        return food_name, confidence_score

    except Exception as e:
        print("Food recognition error:", e)
        return None, 0


# ==========================================
# AUTOMATIC FOOD IMAGE RETRIEVAL - OPENVERSE
# ==========================================

FOOD_IMAGE_FOLDER = 'static/food_images'
os.makedirs(FOOD_IMAGE_FOLDER, exist_ok=True)


def fetch_images(query):
    """
    Find a matching food image using this order:
    1. Reuse a previously downloaded Openverse image.
    2. Reuse an exact matching image in static/uploads.
    3. Search Openverse, check title/tags for relevance, and download it.
    4. Use the existing fallback image if no suitable image is found.

    The original downloaded image bytes are kept unchanged so that images
    with no-derivatives licences are not modified. Licence and attribution
    details are saved in a JSON file beside each downloaded image.
    """

    normalized_query = normalize_food_name(query)

    # The recommendation route adds the word "food" to every search.
    if normalized_query.endswith(' food'):
        normalized_query = normalized_query[:-5].strip()

    safe_name = re.sub(
        r'[^a-z0-9]+', '_', normalized_query.lower()
    ).strip('_')

    fallback_image = url_for('static', filename='uploads/ntn.png')

    if not safe_name:
        return [fallback_image]

    # 1. Reuse only images cached by this relevance-checked version.
    # Older cached images may have been selected without checking their
    # titles/tags, so they are deliberately ignored and replaced.
    try:
        metadata_path = os.path.join(
            FOOD_IMAGE_FOLDER, safe_name + '.json'
        )
        cache_is_verified = False

        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                    cached_metadata = json.load(metadata_file)
                cache_is_verified = cached_metadata.get('relevance_checked') is True
            except (OSError, ValueError, TypeError):
                cache_is_verified = False

        if cache_is_verified:
            for extension in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
                cached_path = os.path.join(
                    FOOD_IMAGE_FOLDER, safe_name + extension
                )
                if os.path.isfile(cached_path):
                    return [url_for(
                        'static',
                        filename=f'food_images/{safe_name}{extension}'
                    )]

        # Remove outdated/unverified cache files so the new search can
        # choose a more relevant result instead of reusing a bad image.
        for extension in ('.jpg', '.jpeg', '.png', '.webp', '.gif'):
            old_cached_path = os.path.join(
                FOOD_IMAGE_FOLDER, safe_name + extension
            )
            if os.path.isfile(old_cached_path):
                os.remove(old_cached_path)

    except OSError as e:
        print('Food image cache check error:', e)

    # 2. Check user-uploaded images for an exact filename match.
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            existing_name = os.path.splitext(filename)[0]
            if normalize_food_name(existing_name) == normalized_query:
                return [url_for('static', filename=f'uploads/{filename}')]
    except OSError as e:
        print('Local image search error:', e)

    # 3. Search Openverse. Search results are filtered by title and tags
    # before downloading so that a restaurant/venue photo is less likely
    # to be selected for a food recommendation.
    try:
        search_url = (
            'https://api.openverse.org/v1/images/?'
            + urllib.parse.urlencode({
                'q': normalized_query + ' food',
                'page_size': 20,
                'license_type': 'commercial'
            })
        )

        search_request = urllib.request.Request(
            search_url,
            headers={'User-Agent': 'CalorieCraft/1.0 (food image search)'}
        )

        with urllib.request.urlopen(search_request, timeout=20) as response:
            search_data = json.loads(response.read().decode('utf-8'))

        results = search_data.get('results', [])

        # Ignore generic words. All remaining food-name terms should be
        # represented in the title/tags for multi-word food names.
        query_terms = [
            word for word in normalized_query.split()
            if len(word) > 2 and word not in {'food', 'dish'}
        ]

        minimum_matches = (
            len(query_terms) if len(query_terms) <= 2
            else max(2, (2 * len(query_terms) + 2) // 3)
        )

        for result in results:
            title = str(result.get('title') or '').lower()
            tags = result.get('tags') or []

            tag_names = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_names.append(str(tag.get('name') or ''))
                else:
                    tag_names.append(str(tag))

            searchable_text = ' '.join([title] + tag_names).lower()
            searchable_text = normalize_food_name(searchable_text)

            matched_terms = sum(
                1 for term in query_terms
                if term in searchable_text.split()
            )

            if query_terms and matched_terms < minimum_matches:
                print('Skipping unrelated Openverse result:', title)
                continue

            image_url = result.get('url')
            if not image_url:
                continue

            # Only download HTTPS image URLs returned by Openverse.
            parsed_image_url = urllib.parse.urlparse(image_url)
            if parsed_image_url.scheme != 'https':
                continue

            try:
                image_request = urllib.request.Request(
                    image_url,
                    headers={'User-Agent': 'CalorieCraft/1.0'}
                )

                with urllib.request.urlopen(image_request, timeout=20) as image_response:
                    image_bytes = image_response.read(8 * 1024 * 1024 + 1)

                # Avoid unexpectedly large downloads.
                if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
                    print('Skipping empty or oversized image:', title)
                    continue

                # Validate the image and identify its real format.
                from io import BytesIO

                with Image.open(BytesIO(image_bytes)) as downloaded_image:
                    image_format = downloaded_image.format
                    downloaded_image.verify()

                extension_by_format = {
                    'JPEG': '.jpg',
                    'JPG': '.jpg',
                    'PNG': '.png',
                    'WEBP': '.webp',
                    'GIF': '.gif'
                }
                extension = extension_by_format.get(image_format)

                if not extension:
                    print('Skipping unsupported image format:', image_format)
                    continue

                image_path = os.path.join(
                    FOOD_IMAGE_FOLDER, safe_name + extension
                )

                # Save the original bytes unchanged. This matters for
                # images published under no-derivatives licences.
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_bytes)

                # Store attribution and licence information for the image.
                attribution_data = {
                    'title': result.get('title'),
                    'creator': result.get('creator'),
                    'license': result.get('license'),
                    'license_version': result.get('license_version'),
                    'license_url': result.get('license_url'),
                    'source_url': result.get('foreign_landing_url'),
                    'image_url': image_url,
                    'relevance_checked': True
                }

                metadata_path = os.path.join(
                    FOOD_IMAGE_FOLDER, safe_name + '.json'
                )
                with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
                    json.dump(attribution_data, metadata_file, indent=4)

                print(
                    'Downloaded Openverse image for:', normalized_query,
                    '| Title:', result.get('title'),
                    '| Licence:', result.get('license')
                )

                return [url_for(
                    'static',
                    filename=f'food_images/{safe_name}{extension}'
                )]

            except Exception as image_error:
                print('Could not download this Openverse image:', image_error)
                continue

    except Exception as e:
        print('Openverse image search error:', e)

    # 4. Use the project's existing fallback image.
    return [fallback_image]


# ==========================================
# CALCULATE BMR
# ==========================================

def calculate_bmr(weight, height, age, gender):

    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    return bmr


# ==========================================
# HOME PAGE
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')


# ==========================================
# ABOUT PAGE
# ==========================================

@app.route('/about')
def about():
    return render_template('about.html')


# ==========================================
# CONTACT PAGE
# ==========================================

@app.route('/contact')
def contact():
    return render_template('contact.html')


# ==========================================
# FOOD RECOMMENDATION PAGE
# ==========================================

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():

    if request.method == 'POST':

        # User input
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])

        gender = request.form['gender']
        allergies = request.form['allergies'].lower()

        # Calculate BMR
        bmr = calculate_bmr(
            weight,
            height,
            age,
            gender
        )

        # Calorie range
        lower_bound = (bmr * 0.9) / 4
        upper_bound = (bmr * 1.1) / 4

        # Filter based on allergies
        if allergies.strip():
            filtered_data = data[
                ~data['Name'].str.lower().str.contains(
                    allergies,
                    na=False,
                    regex=False
                )
            ]
        else:
            filtered_data = data

        # Filter based on calories
        filtered_data = filtered_data[
            (filtered_data['Calories'] >= lower_bound) &
            (filtered_data['Calories'] <= upper_bound)
        ]

        # Select a maximum of five foods
        filtered_foods = filtered_data['Name'].sample(
            n=min(5, len(filtered_data))
        )

        recommendations = []

        for food_name in filtered_foods:

            food_image = fetch_images(food_name + " food")

            food_row = filtered_data[
                filtered_data['Name'] == food_name
            ].iloc[0]

            graph_url = generate_food_graph(
                food_row['Proteins'],
                food_row['Fats'],
                food_row['Carbohydrates']
            )

            recommendations.append({
                'Name': food_name,
                'Type': food_row['Type'],
                'Calories': food_row['Calories'],
                'Proteins': food_row['Proteins'],
                'Fats': food_row['Fats'],
                'Carbohydrates': food_row['Carbohydrates'],
                'FoodImage': food_image[0],
                'GraphURL': graph_url
            })

        return render_template(
            'recommendations.html',
            recommendations=recommendations,
            bmr=round(bmr, 2)
        )

    return render_template('home.html')


# ==========================================
# GENERATE NUTRITION GRAPH
# ==========================================

def generate_food_graph(proteins, fats, carbohydrates):

    labels = [
        'Proteins',
        'Fats',
        'Carbohydrates'
    ]

    values = [
        proteins,
        fats,
        carbohydrates
    ]

    plt.figure(figsize=(5, 3))

    plt.plot(
        labels,
        values,
        marker='o',
        color='b',
        linestyle='-',
        linewidth=2,
        markersize=8
    )

    plt.title('Nutritional Breakdown')
    plt.xlabel('Nutrient')
    plt.ylabel('Amount (g)')
    plt.grid(True)

    img = io.BytesIO()

    plt.savefig(
        img,
        format='png',
        bbox_inches='tight'
    )

    plt.close()

    img.seek(0)

    img_base64 = base64.b64encode(
        img.getvalue()
    ).decode('utf-8')

    img_url = f"data:image/png;base64,{img_base64}"

    return img_url


# ==========================================
# IMAGE UPLOAD AND FOOD RECOGNITION
# ==========================================

@app.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        try:
            # Get uploaded image
            image_file = request.files.get('image')

            if not image_file or not image_file.filename:
                return render_template(
                    'upload_results.html',
                    foods=[],
                    message="Please select an image."
                )

            # Secure filename
            filename = secure_filename(image_file.filename)

            if not filename or '.' not in filename:
                return render_template(
                    'upload_results.html',
                    foods=[],
                    message="Invalid image filename."
                )

            extension = filename.rsplit('.', 1)[1].lower()

            if extension not in ALLOWED_EXTENSIONS:
                return render_template(
                    'upload_results.html',
                    foods=[],
                    message="Please upload JPG, JPEG, PNG or WEBP images."
                )

            # Save uploaded image
            image_path = os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            image_file.save(image_path)

            image_url = url_for(
                'static',
                filename=f'uploads/{filename}'
            )

            # Verify that the uploaded image can be opened.
            with Image.open(image_path) as uploaded_image:
                uploaded_image.verify()

            # Recognize food using the AI model.
            predicted_food, confidence = recognize_food(image_path)

            # Low-confidence prediction
            if predicted_food is None:
                return render_template(
                    'upload_results.html',
                    foods=[],
                    image_url=image_url,
                    detected_food=None,
                    confidence=round(confidence, 2),
                    message=(
                        "Unable to identify this food confidently. "
                        "Please upload a clearer image."
                    )
                )

            # Match recognized food with the nutrition dataset.
            # This function always returns a list.
            matched_foods = match_food_with_dataset(predicted_food)

            # Food recognized, but no nutrition information is available.
            if not matched_foods:
                return render_template(
                    'upload_results.html',
                    foods=[],
                    image_url=image_url,
                    detected_food=predicted_food.replace('_', ' ').title(),
                    confidence=round(confidence, 2),
                    message=(
                        "Food identified, but its nutritional information "
                        "is not available in our dataset."
                    )
                )

            # Return all matching foods.
            # For generic ice cream, this displays both Saffron Ice Cream
            # and Matcha Ice Cream.
            return render_template(
                'upload_results.html',
                foods=matched_foods,
                image_url=image_url,
                detected_food=predicted_food.replace('_', ' ').title(),
                confidence=round(confidence, 2),
                message=(
                    "Food successfully identified! "
                    "Available nutrition matches are shown below."
                )
            )

        except Exception as e:
            print("Error during image processing:", e)

            return render_template(
                'upload_results.html',
                foods=[],
                message=(
                    "An error occurred while processing the image. "
                    "Please try another image."
                )
            )

    return render_template('upload.html')


# ==========================================
# RUN FLASK APPLICATION
# ==========================================

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
