# Automated Badminton Service Fault Detector

## Project Overview

This project automatically detects badminton service faults from video footage using computer vision and deep learning techniques.

The system analyses a badminton serve and detects:

- Service height faults
- Foot contact with the service line
- Foot lifting faults

The pipeline combines:

- Shuttlecock detection
- Foot keypoint detection
- Court line detection
- Hit frame detection
- Homography-based foot movement analysis

## Repository Structure

dataset_preparation/
Scripts used for dataset generation and preprocessing.

training/
Model training notebooks.

pipeline/
Main service fault detection pipeline.

evaluation/
Evaluation scripts used to assess system performance.

development/
Experiments and debugging scripts used during implementation.

Datasets and trained model weights are not included in this repository due to size limitations and participant privacy considerations.
