import numpy as np
from skimage.metrics import variation_of_information

def make_contiguous(array):
    """Make an array contiguous"""
    return np.unique(array, return_inverse=True)[1].reshape(array.shape)

def find_array_difference(arr1, arr2):
    """
    If two labelled arrays have a different number of objects,
    return an array with only the mis-labelled arr1 objects.
    """
    assert arr1.shape == arr2.shape

    # Used for applying original labels to the output array
    arr1_objs = np.unique(arr1)

    # Arrays must be contiguous
    x_objs, x_arr = np.unique(arr1, return_inverse=True)
    y_objs, y_arr = np.unique(arr2, return_inverse=True)
    x_arr = x_arr.reshape(arr1.shape)
    y_arr = y_arr.reshape(arr2.shape)

    out_arr = np.zeros(arr1.shape)

    # Skip background 0 pixels
    for gt_annotation, contig_annotation in zip(x_objs[1:], y_objs[1:]):
        # For a given unique label, find in the indices where this occurs
        same_lab = np.where(y_arr == contig_annotation)

        # On a processed array, check that the unique labels have not been relabelled. 
        if len(np.unique(x_arr[same_lab])) != len(np.unique(y_arr[same_lab])):
            out_arr[same_lab] = gt_annotation
    return out_arr

def intersection_over_union(ground_truth, prediction):
    """Requries input labels to be contiguous"""
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    # A 2D histogram is also known as a heatmap. 
    # Each array is flattened to 1D and then binned into the unique object numbers
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    # Place each pixel into a bin. Number of bins is the number
    # of objects in the image
    # So, if bin 1 has 100 objects within, it indicates that 
    # the object 1 has an area of 100 pixels
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    # Expand dims axis=-1 changes shape from (x,) to (x, 1)
    # This in effect converts a 1D array into a 2D array, with
    # each having just one element. 
    # So, an array: [4, 0, 3] becomes 
    # [[4],
    # [0],
    # [3]]
    area_true = np.expand_dims(area_true, -1)
    # expand_dims axis=0 changes shape from (y,) to (1, y)
    # So array: [4, 0, 3] becomes [[4, 0, 3]] (extra bracket)
    area_pred = np.expand_dims(area_pred, 0)
    # The union is the total area of 
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    IOU = intersection/union
    
    return IOU

def measures_at(threshold, IOU):
    
    matches = IOU > threshold
    
    # For each GT object (from axes=1), check that there is 
    # only one match (ie. one predicted object for one GT object)
    # If np.sum(matches, axis=1) > 1, GT has been over-segmented 
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    official_score = TP / (TP + FP + FN + 1e-9)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    
    return f1, TP, FP, FN, official_score, precision, recall


def get_splits_and_merges(ground_truth, prediction, threshold=0.1):

    # Compute IoU
    IOU = intersection_over_union(ground_truth, prediction)
    
    matches = IOU > threshold
    # Single ground truth object has multiple predictions
    # Was merge
    merge = np.sum(matches, axis=0) > 1
    # Multiple ground truth labels have a single prediction
    # Was split
    split = np.sum(matches, axis=1) > 1

    # r = {"Merges":np.sum(merges), "Splits":np.sum(splits)}
    # results.loc[len(results)+1] = r
    # return results
    return np.sum(merge), np.sum(split)

def evaluate_segmentation(ground_truth, prediction, threshold=0.1):
    """Return a dictionary containing segmentation evaluation"""

    seg_eval = {}

    IOU = intersection_over_union(ground_truth, prediction)

    f1, TP, FP, FN, official_score, precision, recall = measures_at(threshold, IOU)

    # Count objects
    # Exclude background
    true_objects = len(np.unique(ground_truth)[1:])
    pred_objects = len(np.unique(prediction)[1:])

    merge, split = get_splits_and_merges(ground_truth, prediction, threshold)
    merge_rate = (merge / (true_objects - merge)) * 100
    # If an object has been undersegmented, subtract it from the total
    # number of objects
    split_rate = (split / true_objects) * 100

    seg_eval["threshold"] = threshold
    seg_eval["num_gt_objects"] = true_objects
    seg_eval["num_pred_objects"] = pred_objects
    seg_eval["f1"] = f1
    seg_eval["true_positive"] = TP
    seg_eval["false_positive"] = FP
    seg_eval["false_negative"] = FN
    seg_eval["precision"] = precision
    seg_eval["recall"] = recall
    seg_eval["merges"] = merge
    seg_eval["splits"] = split
    seg_eval["perc_merged"] = merge_rate
    seg_eval["perc_split"] = split_rate

    return seg_eval
    # Compute IoU
    # IOU = intersection_over_union(ground_truth, prediction)




