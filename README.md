# Mask_Classifier

In order to run this experiment you'll first need to clone this repository. This repo contains only the code and the testing data.

1. Clone this repository with `git clone https://github.com/GrahamJones95/Mask_Classifier.git` and then enter the repo `cd Mask_Classifier`
2. Download training data from Google Drive from [link](https://drive.google.com/drive/folders/17Fd0uTag6hISmoblUixee6vrOx3GRX6t?usp=sharing "Data Folder")
3. Extract the images into the `Mask_Classifier` folder, there should now be a folder called `my_images`
4. Open a Jupyter notebook instance either by calling `jupyter notebook` from the Terminal or by using a built-in function in an IDE
5. Open and run the notebook mytraining.ipynb. There are quite a few Python libraries which may need to be installed.
Training should take approximately 3 minutes if a GPU is available. After this a ResNet and VGG model will be available for inferencing.
6. To get the accuracy results on an entirely "real" dataset run the real_test.py with the following: 
(If you get CUDA out of memory error you may need to run `nvidia-smi' get the process number taking the majority of the GPU memory and kill it with `kill <Proc #>`
`python real_test.py VGG`
7. To run the model in realtime run: `python webcam_detect.py'
