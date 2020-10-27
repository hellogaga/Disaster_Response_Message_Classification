# A Disaster Message Classification Tool
## Table of contents
- [A Disaster Message Classification Tool](#a-disaster-message-classification-tool)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [File Structure](#file-structure)
  - [How to Use the Package.](#how-to-use-the-package)
  - [Project Motivation](#project-motivation)
  - [Results](#results)
  - [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Installation
The package is based on the following package. The project can be compatible with lower or higher versions of the above packages. However, a detailed test is not carried out. Users might find problems of incompatibilities when executing the code. Please raise a issue when having a problem. <br /> 
- Python: 3.8.5
- pandas: 1.1.1
- numpy:  1.19.1
- scikit-learn: 0.23.2
- pandas: 1.1.3
- plotly: 4.11.0
- flask: 1.1.2
<br /> 

## File Structure
- app
  * template
    * master.html (***main page of web app***)
    * go.html  (***classification result page of web app***)
  * run.py  (***Flask file that runs app***)
- data
  * disaster_categories.csv (***data to process***) 
  * disaster_messages.csv  (***data to process***)
  * process_data.py (***file to process data***)
  * DataResponse.db (***database to save clean data to***)
- models
  * train_classifier.py (***training and saving model***)
  * classifier.pkl (***saved model***) 
- README.md

## How to Use the Package. 
- Step 1: Use ```git clone``` or web download to download all the files into a local place in your machine. 
- Step 2: Navigate to downloaded folder. 
- Step 3: Data Process that cleans the raw data and save a clean database. Can be executed through:
```python
 python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
```
- Step 4: Train the model. Can be executed through:
```python
 python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 
```
- Step 5: Navigate to the folder of ***run.py***
- Step 6: Run the web application. The app will be rendered in **a local host**. Can be executed through:
```python
 python run.py 
```
**NOTE1:** The package is tested on a local machine with **Windows 10** as the operation system. Users with Linux or Mac system might have problems in the file locations. Please go to ***process_data.py*** and ***train_classifier.py*** and revise the file locations accordingly. <br /> 
**NOTE2:** By default, the app will be rendered in a local host. In cloud-based system, such as Udacity's online workspace, revise the line as ```app.run(host='0.0.0.0', port=3001, debug=True)``` in ***run.py***. 

## Project Motivation
This project is motived by the Udacity Data Science Nanodegree project. A disaster-related database was provided by [**Figure Eight**](https://appen.com/). I am supposed to train a model that can classify a coming text message. It is a multiple output classification problem. 

## Results
The main results of the project is the trained model in the 'models' folder. The main outcome of the project is a web application which can be used to classify incoming messages.

## Licensing, Authors, Acknowledgements
The code released subjects to the MIT license. The author appreciate the data provided by the [**Figure Eight**](https://appen.com/) and the code structure from [**Udacity**](https://www.udacity.com/).