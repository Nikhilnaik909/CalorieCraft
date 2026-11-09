# 🍎 CalorieCraft – AI-Powered Diet Planner

**CalorieCraft** is a beta AI-powered diet planning web application developed to help users make informed food choices and maintain a balanced diet. It combines food image recognition, nutritional information, personalized food recommendations, and exercise-related insights in one platform.

This project was independently designed, developed, and implemented as a **solo project**, including the development and training of its food recognition model. Various AI tools were leveraged throughout the development process

## 🚀 Project Overview

CalorieCraft aims to simplify healthy eating by allowing users to explore food information, identify food items through uploaded images, and receive food recommendations based on their personal details and dietary requirements.

The application focuses on making nutritional information more accessible while providing an interactive and user-friendly experience.

## ✨ Key Features

- **Personalized Food Recommendations:** Provides food suggestions based on user information and dietary preferences.
- **Food Image Recognition:** Identifies food items from uploaded images using a machine-learning model.
- **Custom Food Recognition Model:** A food recognition model developed and trained as part of this project.
- **Nutritional Analysis:** Displays nutritional information, including calories, proteins, fats, and carbohydrates.
- **Food Dataset Integration:** Uses structured food datasets to retrieve relevant nutritional information.
- **Food Image Retrieval:** Fetches food images automatically through the Openverse API.
- **Exercise-Related Information:** Includes exercise-related data to support healthier lifestyle choices.
- **Interactive Web Interface:** Provides dedicated pages for recommendations, food information, image uploads, and other project features.

## 🧠 Machine Learning and Model Development

An important component of CalorieCraft is its food recognition functionality.

The food recognition model was developed and trained independently as part of this solo project. It is integrated into the Flask application to process uploaded food images and predict the corresponding food category.

The model is designed to connect image-based food identification with the application's nutritional information and recommendation functionality.

**Model development and integration:** Independently performed as part of CalorieCraft.

## 🛠️ Technology Stack

| Category                | Technologies                                          |
| ----------------------- | ----------------------------------------------------- |
| Programming Language    | Python                                                |
| Backend Framework       | Flask                                                 |
| Frontend                | HTML, CSS                                             |
| Data Processing         | Pandas, NumPy                                         |
| Machine Learning        | Python-based machine-learning and deep-learning tools |
| Model Integration       | Hugging Face ecosystem                                |
| Image Retrieval         | Openverse API                                         |
| Data Storage            | CSV datasets                                          |
| Visualization           | Matplotlib                                            |
| Development Environment | Visual Studio Code                                    |
| Version Control         | Git and GitHub                                        |

## 🔄 Technology Evolution and Improvements

During development, CalorieCraft underwent several technical improvements to make the application more practical, accessible, and maintainable.

| Earlier Approach                 | Updated Approach                            | Purpose                                                                  |
| -------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| Google Custom Search API         | Openverse API                               | Retrieve food images without relying on Google Custom Search credentials |
| AWS Rekognition                  | Hugging Face-based food recognition model   | Use an integrated food recognition model for uploaded food images        |
| External image-search dependency | Automatic image retrieval and local caching | Improve image retrieval and reduce repeated requests                     |

These changes helped move the project toward a more independent and maintainable architecture.

## 📂 Project Structure
```
CalorieCraft/
│
├── app.py
├── test_food_model.py
│
├── dataset/
│   ├── food.csv
│   ├── food_dataset.csv
│   └── food_exercise.csv
│
├── static/
│   ├── css/
│   ├── images/
│   ├── uploads/
│   └── food_images/
│
├── templates/
│   ├── about.html
│   ├── base.html
│   ├── contact.html
│   ├── food_info.html
│   ├── home.html
│   ├── index.html
│   ├── recommendations.html
│   ├── result.html
│   ├── upload.html
│   └── upload_results.html
│
├── .gitignore
└── README.md
```

*Note: Some directories, such as the food image cache and user-uploaded test images, are generated or populated while running the application and are intentionally excluded from Git.*

## ⚙️ Installation and Setup

### 1. Clone the repository
```
git clone https://github.com/Nikhilnaik909/CalorieCraft.git
```

### 2. Navigate to the project directory
```
cd CalorieCraft
```

### 3. Create a virtual environment
```
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**
```
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

If a `requirements.txt` file is available:
```
pip install -r requirements.txt
```

### 6. Run the application
```
python app.py
```

Open the application in your browser at:
```
http://127.0.0.1:5000/
```

*Note: The trained model and any required model files must be available or downloaded according to the application's configuration. Dependency installation instructions should be updated to match the final requirements.txt.*

## 🎯 Project Objectives

- Encourage healthier eating habits through accessible nutritional information.
- Explore the practical application of machine learning in food recognition.
- Connect food identification with nutritional datasets.
- Provide food recommendations through a web-based platform.
- Demonstrate the integration of machine learning, APIs, data processing, and web development in a real-world application.

## 🔮 Future Enhancements

- Voice-assisted interaction.
- Personalized meal and recipe recommendations.
- Automated grocery and shopping lists.
- Advanced dietary and nutritional tracking.
- Smart kitchen integration.
- Improved food recognition accuracy and expanded food categories.

## 👨‍💻 Developer

**Developed independently as a solo project.**

I designed and implemented CalorieCraft, including its web application, food recognition model development and training, dataset integration, API integration, and application functionality.

**GitHub:** [Nikhilnaik909](https://github.com/Nikhilnaik909)

---

*CalorieCraft – Making healthier food choices smarter and simpler.*

