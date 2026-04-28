import torch
from oml.models import ResnetExtractor


def print_available_models():
    print("Available OML ResNet models")

    available_models = list(ResnetExtractor.pretrained_models.keys())
    metric_learning_models = []
    imagenet_models = []

    for model_name in available_models:
        if "moco" in model_name.lower() or "metric" in model_name.lower():
            metric_learning_models.append(model_name)
        else:
            imagenet_models.append(model_name)

    print(f"Total models: {len(available_models)}")
    print(f"Metric learning models: {len(metric_learning_models)}")
    for model_name in metric_learning_models:
        print(f"  - {model_name}")

    print(f"ImageNet models: {len(imagenet_models)}")
    for model_name in imagenet_models:
        print(f"  - {model_name}")


def inspect_moco_model():
    print("\nLoading ResNet50 MoCo v2")

    model_moco = ResnetExtractor.from_pretrained("resnet50_moco_v2")
    print("ResNet50 MoCo v2 loaded")
    print(f"Architecture: {model_moco.arch}")
    print(f"Feature normalization: {model_moco.normalise_features}")
    print(f"Remove FC: {model_moco.remove_fc}")
    print(f"GEM pooling: {getattr(model_moco, 'gem_p', 'None')}")

    test_input = torch.randn(1, 3, 224, 224)
    model_moco.eval()
    with torch.no_grad():
        output = model_moco(test_input)

    print(f"Embedding shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    if hasattr(model_moco.model, "fc"):
        fc_layer = model_moco.model.fc
        print(f"Final layer: {type(fc_layer).__name__}")
        if hasattr(fc_layer, "weight"):
            print(f"FC weight shape: {fc_layer.weight.shape}")
        elif hasattr(fc_layer, "in_features"):
            print(f"FC in features: {fc_layer.in_features}")
            print(f"FC out features: {fc_layer.out_features}")
        else:
            print(f"FC layer: {fc_layer}")

    if hasattr(model_moco.model, "avgpool"):
        pooling = model_moco.model.avgpool
        print(f"Pooling layer: {type(pooling).__name__}")
        print(f"Pooling params: {pooling}")

    return model_moco, test_input


def compare_with_imagenet(model_moco, test_input):
    print("\nComparing with ResNet50 ImageNet")

    model_imagenet = ResnetExtractor.from_pretrained("resnet50_imagenet1k_v1")
    print("ResNet50 ImageNet loaded")

    with torch.no_grad():
        output_moco = model_moco(test_input)
        output_imagenet = model_imagenet(test_input)

    print(f"{'Metric':<25} {'MoCo v2':<15} {'ImageNet':<15}")
    print(f"{'Architecture':<25} {model_moco.arch:<15} {model_imagenet.arch:<15}")
    print(
        f"{'Normalization':<25} {str(model_moco.normalise_features):<15} {str(model_imagenet.normalise_features):<15}"
    )
    print(
        f"{'Remove FC':<25} {str(model_moco.remove_fc):<15} {str(model_imagenet.remove_fc):<15}"
    )
    print(
        f"{'Embedding size':<25} {str(output_moco.shape[1]):<15} {str(output_imagenet.shape[1]):<15}"
    )


if __name__ == "__main__":
    print_available_models()

    model_moco = None
    test_input = None
    try:
        model_moco, test_input = inspect_moco_model()
    except Exception as exc:
        print(f"Failed to load ResNet50 MoCo v2: {exc}")

    if model_moco is not None and test_input is not None:
        try:
            compare_with_imagenet(model_moco, test_input)
        except Exception as exc:
            print(f"Failed to load ResNet50 ImageNet: {exc}")

    print("\nSummary")
    print("1. OML has no pretrained ResNet34 metric learning model.")
    print("2. ResNet50 MoCo v2 is available for metric learning.")
    print("3. ImageNet models can be used as baselines.")
