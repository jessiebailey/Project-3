# Project-3
DS 4002 Project 3

## Software and Platform
The software used for this project is Python. Necessary packages include **OS**, **NumPy**, **TensorFlow**, and **Matplotlib**. The project was developed on both **Windows** and **Mac** platforms.

## Documentation Map
- **DATA Folder:**
  - *Access Data* document explaining how to access the image data, including a download link
  - *Metadata* document summarizing the data, explaining its origin and license, providing an overview, and including two exploratory plots

- **SCRIPTS Folder:**
  - `Project2_eda.py` — code for exploratory data analysis
  - `evaluate_garbage_file.py` — code for the image classification model

- **OUTPUTS Folder:**
  - `EDA1.pdf` - graphical summary of data
  - `EDA2.pdf` - written summary of data
  - `Accuracy_Loss_CNN.png` - graphs of the validation and training accuracy and loss for the Convolutional Neural Network (CNN) model
  - `Accuracy_Loss_MobileNetV2.png` - graphs of the validation and training accuracy and loss for the MobileNetV2 model
  - `MobileNetV2_Summary.png` - output summary of overall model accuracy for MobileNetV2, with a chart of which types of images were classified as being recyclable, compostable, or just trash

- **LICENSE:** MIT license file for the repository  
- **README:** Provides an overview of the repository and its contents

## Instructions for Reproduction
1. Follow the link in the *Access Data* document to download the necessary image data. Save it to a known location on your computer.
2. Install all required Python packages listed above.
3. Run `Project2_eda.py` to perform exploratory data analysis. A plot and descriptive summary will be generated.
4. Run `evaluate_garbage_file.py` to classify each image as **recycling**, **compost**, or **trash**. Accuracy and loss graphs for both the CNN and MobileNetV2 models will be generated, as well as a short output summary for the MobileNetV2 model.
