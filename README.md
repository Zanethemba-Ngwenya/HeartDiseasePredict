# Heart Disease Prediction

## Project Overview

The Heart Disease Prediction project utilizes machine learning techniques to predict the presence of heart disease based on various patient attributes. The project employs the k-Nearest Neighbors (k-NN) algorithm from the Weka library for classification and uses data visualization to analyze age distribution among the patients.

### Key Features

- **Data Loading**: The project reads heart disease data from a CSV file.
- **Data Preprocessing**: Normalization of features to improve model accuracy.
- **Model Training**: Utilizes the k-NN algorithm to classify the data into healthy and unhealthy categories.
- **Model Evaluation**: Calculates and displays the accuracy of the model based on a test dataset.
- **Data Visualization**: Plots the age distribution of patients to provide insights into the dataset.

### Technologies Used

- **Java**: The primary programming language for the application.
- **Weka**: A popular machine learning library used for data mining tasks.
- **XChart**: A Java library for creating charts and visualizations.
- **Maven**: A build automation tool for managing project dependencies and builds.

### Requirements

To run this project, you need:

- Java Development Kit (JDK) 1.8 or higher
- Apache Maven

### Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/HeartDiseasePredict.git
   cd HeartDiseasePredict


2. **Build**:
    ```mvn clean install

4. **Run application**:
   ```mvn exec:java -Dexec.mainClass="wethinkcode.HeartDiseasePrediction"
