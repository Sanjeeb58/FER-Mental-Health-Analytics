@echo off
echo Activating Virtual Environment...
cd /d "C:\Users\HP\Desktop\Facial-Expression-Recognition-FER-for-Mental-Health-Detection-"
call venv\Scripts\activate.bat
echo Running Webcam Emotion Detection...
python utilities\webcam_face_emotion.py
pause
