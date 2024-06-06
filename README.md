# Emotional Arousal Identification in VR Environment using Eye Tracking

## Objective:
Identifying emotional arousal in the VR environment by clustering eye-tracking features contributes to understanding VR's Sense of Presence (SoP).

## Introduction:
The VR environment aims to provide immersive experiences for users. While interaction design is crucial, measuring immersion is implicit. Sense of Presence (SoP) in VR is subjective and involves emotional, cognitive, and behavioural responses. This project focuses on emotional responses in VR, particularly through eye-tracking features, as indicators of SoP.

## Literature Survey:
Existing experiments have studied emotional arousal and valence in VR using physiological signals like PPG, ECG, EEG, and eye metrics. Machine Learning and Deep Learning models, particularly Support Vector Machines, have been employed for classification. The circumplex model of affect is commonly used. Pupil dilation, indicating emotional arousal, is a significant factor.

## Method:
Participants underwent a VR scenario, and eye-tracking features (e.g., blink rate, pupil dilation) were extracted. Data was divided into windows, and clustering algorithms were used for classification. The correlation between blink rate and pupil dilation was analyzed. Training and testing data sets were utilized.

## Results:
- Plots of normalised pupil dilation data and window classification.
- Clustering of data and class prediction for different windows of the immersive VR experience.
- Results from the test data.

## Inclusion of VR Concepts:
Eye Tracking, Sense of Presence, Evaluation of VR, Immersion.

## Discussion:
The model classified data based on pupil dilation, indicating its significance in emotional arousal detection. Consistent classification between training and testing data supports the existence of a pattern in pupil dilation, indicative of emotional arousal and SoP in VR.

## Conclusion:
Emotional arousal in VR can be measured through eye tracking, highlighting the potential for emotion detection models. The interdisciplinary applications of such models are extensive.

## Acknowledgments:
Special thanks to Lokeshwaran for providing data and support for the project execution.

## Repository Contents:
- **ED19B005_AM5011_Project.py/**: Contains the code files.
- **ED19B005_ProjectReport.pdf/**: Contains the project report.
- **README.md**: This README file.
