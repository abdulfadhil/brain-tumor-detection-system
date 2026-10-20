# About the Model

The trained model is not included in this GitHub repository because the file is about 1.18 GB in size.

## Which model file is needed?

The Flask application looks for the model at this location:

`models/brain_tumor_cnn.h5`

To use the prediction feature, place a compatible model file at this path.

Not every `.h5` file will work. The model's architecture and input shape must match what the application expects in `app.py`.

## Where can I get the model?

There is currently no public download link for the model. Use only a model that you have permission to use and that is compatible with this project.

## Limitations

This is an academic project for learning and research. The model's output is experimental and has not been clinically validated. It must not be used to diagnose a medical condition or make treatment decisions.