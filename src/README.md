# Source Code Guide

This directory contains the Python modules used by the Brain MRI Analysis project, along with separate training and prediction experiments.

## Main Application Modules

- `mri_input_validation.py` - Validates the required MRI files and checks image dimensions and alignment.
- `mri_3d_visualization.py` - Generates the interactive 3D visualization used by the Flask application.
- `train_3d_cnn.py` - Training script for an experimental binary 3D CNN. Requires a configured MRI dataset and suitable TensorFlow environment.

## Experimental and Standalone Scripts

- `brain_3d_visualization.py` - Standalone MRI visualization experiment.
- `brain_tumor_training_experiment.py` - Separate 3D CNN training experiment.
- `multiclass_segmentation_experiment.py` - Experimental multiclass segmentation pipeline.
- `tumor_prediction.py` - Legacy standalone prediction script.
- `tumor_prediction_candidate.py` - Candidate standalone binary prediction script.
- `tumor_analysis_experiment.py` - Experimental tumor analysis script.

## Important Notes

- The standalone experiments are not necessarily part of the Flask application's execution path.
- Different scripts may use different preprocessing methods, model architectures, input paths, and output filenames.
- A model produced by one script must not be assumed compatible with the Flask application without checking its architecture and preprocessing requirements.
- Model outputs and mask-based measurements are experimental and have not been clinically validated.
- These scripts are provided for educational and research purposes, not medical diagnosis or treatment decisions.
