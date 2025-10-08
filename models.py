import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_large, ConvNeXt_Large_Weights, resnet50
import timm


def build_convnext_tiny_model(num_classes=498):
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[2].in_features, num_classes)
    )
    return model


def build_resnet50_model(num_classes=498):
    model = resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_efficientnetv2_model(num_classes=498):
    model = timm.create_model('efficientnetv2_rw_m', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

def build_efficientnetv2_l_model(num_classes=498):
    model = timm.create_model('tf_efficientnetv2_l', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

def build_efficientnetv2_xl_model(num_classes=498):
    model = timm.create_model('tf_efficientnetv2_xl', pretrained=True)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(1024, num_classes)
    )
    return model

def build_convnext_large_model(num_classes=498):
    weights = ConvNeXt_Large_Weights.DEFAULT
    model = convnext_large(weights=weights)

    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[2].in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(1024, num_classes)
    )
    return model
