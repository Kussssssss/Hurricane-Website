# Deployment Guide for Hurricane Trajectory Analysis Application

This guide provides instructions for deploying the Hurricane Trajectory Analysis application to various hosting platforms for permanent access.

## Option 1: Streamlit Cloud (Recommended)

Streamlit Cloud is the easiest way to deploy Streamlit applications with free hosting.

### Prerequisites
- GitHub account
- Fork or upload this repository to your GitHub account

### Deployment Steps

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch (main/master), and the main file path (`app_enhanced.py`)
5. Click "Deploy"

Your app will be deployed with a permanent URL like `https://username-app-name-streamlit-app.streamlit.app`

## Option 2: Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Deployment Steps

1. Create a new Heroku app:
```
heroku create hurricane-trajectory-app
```

2. Push the repository to Heroku:
```
git push heroku master
```

3. Ensure at least one instance is running:
```
heroku ps:scale web=1
```

4. Open the app:
```
heroku open
```

## Option 3: Google Cloud Run

### Prerequisites
- Google Cloud account
- Google Cloud SDK installed

### Deployment Steps

1. Create a new project or select an existing one:
```
gcloud projects create hurricane-trajectory-app
gcloud config set project hurricane-trajectory-app
```

2. Enable required APIs:
```
gcloud services enable cloudbuild.googleapis.com run.googleapis.com
```

3. Build and deploy the container:
```
gcloud builds submit --tag gcr.io/hurricane-trajectory-app/hurricane-app
gcloud run deploy hurricane-app --image gcr.io/hurricane-trajectory-app/hurricane-app --platform managed
```

## Option 4: GitHub Pages with Streamlit Static Export

For a static version of the application:

1. Install streamlit-static:
```
pip install streamlit-static
```

2. Export the app:
```
streamlit-static export app_enhanced.py
```

3. Push the exported files to a GitHub repository
4. Enable GitHub Pages in the repository settings

## Local Deployment

To run the application locally:

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the application:
```
streamlit run app_enhanced.py
```

## Docker Deployment

### Prerequisites
- Docker installed

### Deployment Steps

1. Build the Docker image:
```
docker build -t hurricane-app .
```

2. Run the container:
```
docker run -p 8501:8501 hurricane-app
```

3. Access the application at `http://localhost:8501`

## Deployment Files

The repository includes several configuration files for different deployment options:

- `requirements.txt`: Python dependencies for pip
- `environment.yml`: Conda environment configuration
- `Procfile`: Configuration for Heroku
- `app.yaml`: Configuration for Google App Engine
- `.streamlit/config.toml`: Streamlit configuration

## Post-Deployment

After deploying your application:

1. Test all functionality to ensure it works as expected
2. Update the URL in the README.md file
3. Share the URL with your users

## Support

If you encounter any issues during deployment, please refer to the documentation of the respective platform or open an issue in the repository.
