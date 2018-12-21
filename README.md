# AnomalyDetection

Problem: Build a method (unsupervised or supervised) that classifies malformed shipments, according to whether they have incorrect data, missing data, or abnormal from typical shipments in other ways, etc.

Hypothesis:

It is expected that missing values, NaN values, and incorrect data will show up in a cluster group.
The unsupervised modeling will work by grouping those rows with the above instances, due to their similarity.
The model can be assessed by evaluating the within group sum of squares to the between group sum of sqaures.
An ideal ratio would be a tight WGSS and a broad BGSS that will be evident of an accurate anomaly detection.
