from gesund.metrics.classification.create_validation import ValidationCreation as ClassificationValidationCreation
from gesund.metrics.semantic_segmentation.create_validation import ValidationCreation as SegmentationValidationCreation
from gesund.metrics.object_detection.create_validation import ValidationCreation as ObjectDetectionValidationCreation

def get_validation_creation(problem_type):
    if problem_type == 'classification':
        return ClassificationValidationCreation
    elif problem_type == 'semantic_segmentation':
        return SegmentationValidationCreation
    #elif problem_type == 'instance_segmentation':
    #    return SegmentationValidationCreation
    elif problem_type == 'object_detection':
        return ObjectDetectionValidationCreation
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
