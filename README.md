# Next-Gen Personalized E-commerce Assistant

<img width="1914" height="971" alt="image" src="https://github.com/user-attachments/assets/16dba696-4794-4812-aff1-d403d6f668f6" />

<img width="1525" height="916" alt="image" src="https://github.com/user-attachments/assets/35dc1edd-31d6-45dd-b2ab-09bf1d81a1b8" />



This project is a Streamlit web application that provides personalized product recommendations. It uses computer vision models to predict a user's age, gender, and ethnicity from an image and then leverages a large language model (LLM) based agent to suggest products tailored to their demographic profile and preferences.

## Features

- **Demographic Prediction**: Predicts age, gender, and race from a user's image.
- **Personalized Recommendations**: Utilizes an AI agent to provide product recommendations based on demographics and user input.
- **Web-Based Interface**: An interactive and easy-to-use web interface built with Streamlit.
- **Flexible LLM Backends**: Supports multiple LLM providers, including Openrouter and Grok.
- **Privacy Conscious**: No user images or personal data are stored. All processing is temporary.
- **Containerized**: Includes a Dockerfile for easy setup and deployment.

## How It Works

1.  **Image-based Prediction**: The user uploads an image or uses their camera. The application detects and crops the face from the image.
2.  **Demographic Analysis**: Pre-trained machine learning models predict the user's age, gender, and race based on the facial features.
3.  **User Preferences**: The user can specify their budget, location, and desired product category.
4.  **AI-Powered Recommendations**: A LangChain agent is provided with the demographic information and user preferences.
5.  **Web Search**: The agent uses the Tavily API to search for relevant products on popular e-commerce websites.
6.  **Product Display**: The recommended products are displayed in the application with their name, description, price, and an image fetched from the web.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/product_recommendation_with_agent.git
    cd product_recommendation_with_agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Keys:**
    You will need to provide API keys for the LLM provider and the Tavily search API. The application will prompt you to enter these in the sidebar. You will also need a MotherDuck token for the `main.py` script.

## Usage

### Streamlit Application

To run the web application, use the following command:

```bash
streamlit run src/project_folder/app.py
```

Once the application is running, you can:

1.  Select your preferred LLM provider (Openrouter or Grok) and enter your API keys in the sidebar.
2.  Upload a photo or use the camera to capture an image.
3.  Click the "Run" button to see the predicted age, gender, and race.
4.  Click the "Recommend" button to proceed to the recommendation section.
5.  Select a product category, budget, and location.
6.  Click "Go" to get personalized product recommendations.

### Command-Line Interface (CLI)

You can also use `main.py` to run predictions from the command line.

**Single Prediction:**

To predict a specific attribute (age, gender, or race) from a single image:

```bash
python src/project_folder/main.py --prediction_type single --target <target> --image_path <path_to_image>
```

-   `<target>`: Can be `age`, `gender`, or `race`.
-   `<path_to_image>`: The path to the image file.

**Example:**

```bash
python src/project_folder/main.py --prediction_type single --target age --image_path /path/to/your/image.jpg
```

**Batch Processing:**

The script can also be used for batch processing, training, and evaluation. Use the `--prediction_type batch` flag for this.

## Dockerization

You can build and run the application using Docker.

1.  **Build the Docker image:**
    ```bash
    docker build -t product-reco-agent .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8501:8501 -e MOTHERDUCK_TOKEN="YOUR_TOKEN" -e OPENROUTER_API_KEY="YOUR_KEY" -e TAVILY_API_KEY="YOUR_KEY" product-reco-agent
    ```
    This will start the Streamlit application on port 8501. You will need to pass your API keys as environment variables.

## Project Structure

```
├───.github/            # CI/CD workflows
├───src/
│   └───project_folder/
│       ├───app.py      # Main Streamlit application
│       ├───main.py     # Main script for prediction
│       ├───recommendation_agent.py # LangChain agent for recommendations
│       ├───models/     # Trained models for age, gender, and race prediction
│       ├───scripts/    # Scripts for data loading, embeddings, and training
│       └───...
├───Dockerfile
├───requirements.txt
└───README.md
```

## Key Dependencies

This project relies on several key libraries:

-   **Streamlit**: For creating the web application.
-   **LangChain**: For building the recommendation agent.
-   **OpenCV**: For image processing and face detection.
-   **TensorFlow/Keras**: For the demographic prediction models.
-   **Tavily**: For the product search API.
-   **CatBoost, LightGBM, Scikit-learn**: For the machine learning models.
-   **Docker**: For containerization.
-   **uv**: For python packaging.

## CI/CD

This project is configured with CI/CD pipelines for continuous integration and deployment. The workflow files can be found in the `.github/workflows` directory.
