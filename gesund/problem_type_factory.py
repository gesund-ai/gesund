from gesund.metrics.classification.create_validation import ValidationCreation as ClassificationValidationCreation
from gesund.metrics.semantic_segmentation.create_validation import ValidationCreation as SegmentationValidationCreation
from gesund.metrics.object_detection.create_validation import ValidationCreation as ObjectDetectionValidationCreation

def get_validation_creation(problem_type):
    """
    Retrieve the appropriate validation creation class based on the given problem type.
    
    This function returns a specific `ValidationCreation` class depending on the type of problem 
    (e.g., classification, semantic segmentation, or object detection). The returned class is used 
    to create validation logic for that particular problem type.

    :param str problem_type: 
        The type of machine learning problem. Must be one of the following:
        - 'classification': For classification tasks.
        - 'semantic_segmentation': For semantic segmentation tasks.
        - 'object_detection': For object detection tasks.
    
    :raises ValueError: 
        If the `problem_type` is unknown or not supported.
    :return: 
        A `ValidationCreation` class specific to the given problem type.
    :rtype: type

    """

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
