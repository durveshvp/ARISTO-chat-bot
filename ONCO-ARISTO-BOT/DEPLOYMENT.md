<!-- # Onco-Med ChatBot Deployment Guide

This document provides instructions to deploy your Onco-Med ChatBot to the internet using Render.com.

## Deployment to Render.com

### Step 1: Create a Render Account
1. Go to [Render.com](https://render.com/) and sign up for a free account

### Step 2: Deploy the Application
1. From your Render dashboard, click on "New" and select "Web Service"
2. Connect your GitHub account or choose "Upload Files" if you don't have Git set up
3. If uploading files, zip your entire project folder and upload it
4. Configure the service:
   - Name: onco-aristo-bot (or your preferred name)
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true`
   - Select the free plan option

### Step 3: Set Environment Variables
1. In your service settings, go to the "Environment" tab
2. Add the following environment variable:
   - GOOGLE_API_KEY: [Your Google API Key]

### Step 4: Deploy
1. Click "Create Web Service" and wait for the deployment to complete
2. Once deployed, Render will provide you with a URL to access your application

## Alternative: Deploy to Streamlit Cloud

If you prefer using Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign up and link your GitHub account
4. Select your repository and deploy
5. Add your GOOGLE_API_KEY as a secret in the Streamlit Cloud settings

## Important Notes

- Make sure your GOOGLE_API_KEY is kept secure and never committed to version control
- The free tier of Render may have some limitations in terms of compute resources
- Your application will sleep after periods of inactivity on the free tier
- For production use, consider upgrading to a paid plan

## Troubleshooting

If you encounter issues during deployment:
- Check the build logs for any error messages
- Ensure all dependencies are correctly listed in requirements.txt
- Verify your environment variables are set correctly -->
