## Deepfake Detector Application 

This is a web application built with Streamlit for detecting images using a deep learning pre-trained model.  

# Model 

The trained model used for image classification is stored in the my_best_model_fastai.pkl file. This model was trained on ResNet50, adopting the Fastai library and achieved high accuracy in detecting deepfake art images. 

Requirements 

To run the web application, you need to have the following dependencies installed. You can install them using the provided requirements.txt file in the directory. 

Prerequisites 

Python 3.x 

Streamlit  

specified in requirements.txt 

 

Running the Web Application 

Clone the repository: 

“git clone https://github.com/femy4ever/Deepfake-Art-Image-Detector.git” 

Navigate to the project directory: 

“cd Deepfake-Art-Image-Detector" 

Install dependencies: 

“pip install -r requirements.txt” 

Create a new virtual environment and activate it: 

“python3 -m venv myenv”  

Then; 

“source myenv/bin/activate” 

Run the streamlit application: 

“streamlit run app.py” 
	The application will start locally, and you can access it in your web browser by visiting 	http://localhost:8501. 

Usage 

To upload an image using the provided format (.jpg, .jpeg, .png, .gif, .webp). 

Click the "Upload" button to classify the loaded image. 

The result will display whether the image is AI (Artificial Intelligence) generated or Real, with a confidence level. 

 

Basic File Structure. 
├── app.py                # Streamlit web application script 
├── my_best_model_fastai.pkl  # Pre-trained model file 
├── requirements.txt      # All Dependencies 
└── README.md             # You're reading this eh! 

Contributing to the work 

Contributions are welcome! Feel free to open issues or pull requests. 

 

 