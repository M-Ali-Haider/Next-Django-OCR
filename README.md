# Next-Django-OCR Project

## Project Overview
This project combines a Django backend with a Next.js frontend, featuring OCR (Optical Character Recognition) functionality.

## Prerequisites
- Python 3.x
- Node.js
- pip
- npm

## Getting Started

### Cloning the Project
bash
git clone https://github.com/M-Ali-Haider/Next-Django-OCR.git
cd Next-Django-OCR


### Backend Setup
1. Navigate to the backend directory
bash
cd backend


2. Install Python dependencies
bash
pip install -r requirements.txt


3. Run database migrations
bash
python manage.py migrate


4. Start the backend server
bash
python manage.py runserver

The backend server will typically run on http://127.0.0.1:8000

### Frontend Setup
1. Create a .env file in the frontend directory
2. Add the following to the .env file:

NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000

*Note:* Ensure the URL does not end with a slash /

3. Install frontend dependencies
bash
npm install


4. Start the frontend development server
bash
npm run dev

The frontend will typically run on http://localhost:3000

## Troubleshooting
- Ensure all dependencies are correctly installed
- Check that your backend server is running before starting the frontend
- Verify the .env file configuration