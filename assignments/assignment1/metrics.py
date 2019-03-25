def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for p, gt in zip(prediction, ground_truth):
        if p and gt:
            true_positives += 1
        elif p:
            false_positives += 1
        elif gt:
            false_negatives += 1
        else:
            true_negatives += 1

    if true_positives or false_positives:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 1.0

    if true_positives or false_negatives:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 1.0

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    if recall or precision:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return (prediction == ground_truth).sum() / len(prediction)
