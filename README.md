# Background
Q1. Why bees are important?
- A third of the world's food production depends on bees.
- Major pollinators.

Q2. Problem
- Declining bee population.
- Mites and ants infestation.
- Frequent check-ups on the hive is desirable to monitor the hive strength and health.
- Manual investigation of beehives is time intensive and often distrupts the beehive environment.

# Motivation & Objective
How can we help ?
- Improve the ability to understand the hive health without checking for the hive manually.
- Devise a non-invasive hive health checkup technique using ML.
- Studies show that honeybee images flying in/out of the hive can be used to draw inferences about the hive health.

# Contribution
- Use convolutional neural network (CNN) to design ML models to predict hive health.
- Assess the models for their performances.

# Dataset Description
- Our dataset contains 5100+ bee images annotated with location, date, time, subspecies, health condition, caste, and pollen.
- Dataset link: https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images

# Methods & Implementation
Main strategies of our project

To classify our images we use a convolutional neural network (CNN) as it is the most popular deep learning model for image classification, and past research has shown that it can far outperform other models. The TensorFlow library was used to develop the network. We tested our data using two CNN models. 

The first prototype is a CNN written from scratch. It contains two convolutional layers. Each layer is followed by a max pooling layer. The first convolutional layer has 32 filters with kernel sizes of 3x3 and a rectified linear unit activation. The second convolutional layer is similar to the first, except it has 64 filters. Each max pooling layer reduces the dimensionality of the previous layer by two. The output layer has six units, which corresponds to the six labels for beehive health. Softmax activation is used to output prediction probability for each label. The model contained 1,223,622 trainable parameters. 

For the second CNN, we use a pretrained model. TensorFlow’s mobile net is chosen as it is a relatively small and efficient CNN2. To make the model suitable (as the model was originally trained for 1,000 different classes) the last six layers were modified so that there were six output classes. In total the model has 3,213,126 trainable parameters.  
 
The entire dataset was divided into training, validation, and testing roughly in the ratio of 70:20:10. Parameters were kept the same across both models for consistency in comparison (Table 1).

Parameter
Value
Batch Size
10
Epochs
10
Learning Rate
0.0001
Loss function
Categorical cross-entropy

Table 1. CNN parameters. 

The code was run from Google Colaboratory using GPU mode. For each epoch, the training time varied from 5 to 39 seconds for the first model and from 22 to 24 seconds for the pretrained CNN. Since this was our first attempt towards developing a model using CNN, we referenced a tutorial3 as guidance.

# Results
![bee_results_CM](https://user-images.githubusercontent.com/82466266/234068926-5a3dfad1-300a-450b-81c5-cb31966e3ea6.JPG)
![bee_performance_measures](https://user-images.githubusercontent.com/82466266/234068967-cf598347-c252-4604-bf5e-d91cb72445e9.JPG)
![image](https://user-images.githubusercontent.com/82466266/234068782-9b4992eb-0837-48b9-be52-d713daa523a8.png)
Based on a given bee (input) image, our classifiers would predict the status of the hive. There were six predicted labels defined earlier. Some 3000+ images were used for training and an accuracy of ~99% was achieved for both the CNN models. On the validation set we were able to reach an accuracy of ~86% and 88% for the initial model and on the pretrained model, respectively. We then derived the confusion matrix for these multilabel classifiers and used the matrix to calculate precision, recall, f1 measure, receiver operating characteristic (ROC) curve, and area under curve (AUC) on the test dataset. We use the one-vs-the-rest(OvR) multiclass strategy to derive the ROC and AUC.

On the test dataset, our initial model achieved an accuracy of 86% whereas after using the pre-trained model our classifier had 96% accuracy. In the initial model, we observe a unique pattern from the confusion matrix (Figure 7). The vast majority of incorrect predictions for the label ‘HiveBeingRobbed’ belong to the class label ‘Healthy’ and vice-versa. This indicates that data for these two class labels may have some overlapping attributes as a result of which the model is unable to differentiate the two labels. Looking back into the definitions of a robber bee (which cater to ‘HiveBeingRobbed’ status) and healthy bee, we make the below hypothesis.

Hypothesis: Robber bees usually fly towards a hive to destroy the hive and steal any stored nectar. These robber bees have shiny bodies with no pollen. On the other hand, healthy bees when leaving the hive also have shiny bodies and do not have any pollen. They will only have pollen on their bodies when they return to the hive after collecting nectar. Therefore, it is likely for our model to incorrectly distinguish between robber bees and healthy bees since the input data set does not contain any information about the direction with respect to the hive. That is, we cannot conclude whether a bee is flying into or out of the hive which would be essential to differentiate between robber bees flying towards a hive and healthy bees flying away from the hive.

The first model performs poorly for the ‘HiveBeingRobbed’ label. The same is reflected in the ROC and AUC for this label where we see that ‘HiveBeingRobbed’ has the lowest AUC (Fig 9). After using the pretrained model we do not see any significant improvement for this label (refer to the recall, f1-score values from Fig 8). However, the performance for the other class labels improves drastically. This implies that the pretrained model was able to improve the overall image classification problem and also strengthens our hypothesis that the issue with ‘HiveBeingRobbed’ is at the input data level and not with the model.



# Discussion
This model improves the ability to understand the hive health without checking the hive manually.
Using the pretrained CNN, the final model has 96% accuracy on test data.


Cons 

Dataset currently has no way to identify the direction of flight of the bees - towards or away from the hive. The direction of flight of bees helps to identify the difference between the robber and healthy bees.
The original batch of images was extracted from still time-lapse videos of bees and each frame of the video was subtracted against that background to bring out the bees in the forefront. As we are not bee experts, we were unable to tell the difference in the photos as image quality is inconsistent. It would be beneficial to collaborate with apiarists to help us in this.
As CNN’s are black-box models, the impact of different attributes on classification probabilities is unknown. Therefore, it is not known whether our models are making predictions based on the bee itself or some other attributes in the background. 
The dataset is biased because the source of varroa images is from the same location. Since they are from the same location, the model might train based on the other common feature of the image from the background rather than the bee.
MissingQueen data has very few images, and all are most likely for the same hive, therefore the model could be learning based on the image background.

Conclusion 

The first prototype of our CNN model presented during the presentation fell short of achieving the desired results and was giving an accuracy of 86% on the test data set. During our demonstration, we proposed to use additional techniques to try and improve the model’s performance. Following that route, we tested our classifier by adding a pre-trained model for comparison. This bumped up the overall model’s performance by improving both the validation and test data results. We were able to achieve an accuracy of 96% on the test data.
However, we are still struggling to achieve proper results for the ‘HiveBeingRobbed’ class label. This would probably require gathering more information on how to identify robber bees and enhance our input data set to accurately capture/identify those features.
In conclusion, while there are limitations to the dataset and machine learning model used in this project (as specified in the cons section above), it does have the potential to have a large impact in the apiarist community. An accurate prediction model could greatly reduce the hive maintenance time for beekeepers and constant automatic monitoring of the hive would ensure the beekeeper could act quickly when there is a probability that the hive is unhealthy. Additionally, the collection of this data could lead to further research and discovery in the life of bees. For example, the effect of climate and seasonality could also be monitored and analysed.
Future Directions

Currently, we manually divided the dataset into the train/validate/test data sets. In the future, we would like to shuffle dynamically to ensure our model renders the same level of performance and also investigate wrongly classified images and their metadata to find possible correlations.
It would be beneficial to gather more images from a variety of hives across the United States, this would decrease the potential bias in our dataset for classification based on the image background. 

# References
- https://www.kaggle.com/jenny18/honey-bee-annotated-images
- https://arxiv.org/abs/1704.04861
- https://deeplizard.com/learn/video/RznKVRTFkBY
