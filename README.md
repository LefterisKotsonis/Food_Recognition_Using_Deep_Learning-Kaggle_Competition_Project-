# Food_Recognition_Using_Deep_Learning-Kaggle_Competition_Project
This project was developed as an assignment for my MSc programme in Data Science. Kaggle was used for an in-class competition focused on multi-label food recognition. The dataset contains 40,000 training and 1,000 testing food images across 498 classes. The goal was to classify all food items present in an image and submit predictions evaluated by the micro F1-score metric.

# Features

### Multiple model architectures:

ResNet50, EfficientNetV2-M, EfficientNetV2-L, EfficientNetV2-XL, ConvNeXt-Tiny, and ConvNeXt-Large.

#### Implemented transfer learning and fine-tuning.

#### Automated training, validation, and early stopping pipelines.

#### Threshold tuning for multi-label optimization.

#### Model ensembling for final submission.

# Results - Best Model 

Model (Ensemble Method): ConvNeXt-L + EffNetV2-L + EffNetV2-XL

Micro F1 Score: 0.50088

