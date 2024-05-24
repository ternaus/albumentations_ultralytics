import cv2
import numpy as np
import torch
import torchvision


def cxcywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from format (x_center, y_center, width, height)
    to format (x1, y1, x2, y2).
    """
    y = x.clone()

    width = x[:, 2]
    height = x[:, 3]
    y[:, 0] = x[:, 0] - width / 2  # x1
    y[:, 1] = x[:, 1] - height / 2  # y1
    y[:, 2] = x[:, 0] + width / 2  # x2
    y[:, 3] = x[:, 1] + height / 2  # y2
    return y


def xyxy2cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from format (x1, y1, x2, y2) to format (x_center, y_center, width, height).

    Args:
        x (torch.Tensor): A tensor of shape (n, 4) containing bounding boxes in format (x1, y1, x2, y2).

    Returns:
        torch.Tensor: A tensor of shape (n, 4) containing bounding boxes in format (x_center, y_center, width, height).
    """
    y = x.clone()

    width = x[:, 2] - x[:, 0]
    height = x[:, 3] - x[:, 1]
    y[:, 0] = x[:, 0] + width / 2  # x_center
    y[:, 1] = x[:, 1] + height / 2  # y_center
    y[:, 2] = width  # width
    y[:, 3] = height  # height
    return y


def extract_and_filter(
    prediction: torch.Tensor, conf_thres: float
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Extract boxes, scores, and class labels from predictions and filter by confidence threshold.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4, num_boxes)
        conf_thres (float): Confidence threshold for filtering boxes.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: Lists of boxes, scores, and classes.
    """
    batch_size = prediction.shape[0]

    boxes, scores, class_labels = [], [], []

    for i in range(batch_size):
        pred = prediction[i]

        # Extract boxes (first 4 columns), scores (next num_classes columns)
        b = pred[:4, :].T
        s = pred[4:, :].T

        # Filter by confidence threshold
        max_scores, labels = s.max(dim=1)
        mask = max_scores > conf_thres

        boxes.append(b[mask])
        scores.append(max_scores[mask])
        class_labels.append(labels[mask])

    return boxes, scores, class_labels


def non_max_suppression(
    prediction: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 300, max_wh: int = 7680
) -> list[torch.Tensor]:
    """Perform non-maximum suppression (NMS) on a set of boxes for a batch of predictions.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4, num_boxes)
        conf_thres (float): Confidence threshold below which boxes will be filtered out.
        iou_thres (float): IoU threshold below which boxes will be filtered out during NMS.
        max_det (int): Maximum number of boxes to keep after NMS.
        max_nms (int): Maximum number of boxes into torchvision.ops.nms().
        max_wh (int): Maximum box width and height in pixels.

    Returns:
        List[torch.Tensor]: A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6) containing the kept boxes, with columns (x1, y1, x2, y2, confidence, class).
    """
    boxes, scores, class_labels = extract_and_filter(prediction, conf_thres)
    output = []

    for i in range(len(boxes)):
        if len(boxes[i]) == 0:
            output.append(torch.zeros((0, 6), device=prediction.device))
            continue

        # Convert from xywh to xyxy
        boxes_i = cxcywh2xyxy(boxes[i])

        # Convert to fp32 for NMS
        boxes_i = boxes_i.float()
        scores_i = scores[i].float()
        class_labels_i = class_labels[i].float()

        # Adjust box coordinates for batched NMS
        offsets = class_labels_i * max_wh
        boxes_i += offsets[:, None]

        # Perform batched NMS
        keep = torchvision.ops.batched_nms(boxes_i, scores_i, class_labels_i, iou_thres)
        keep = keep[:max_det]

        # Revert the offsets after NMS
        boxes_i -= offsets[:, None]

        result = torch.cat((boxes_i[keep], scores_i[keep, None], class_labels_i[keep, None]), dim=1)
        output.append(result.half() if prediction.dtype == torch.float16 else result)

    return output


def plot_boxes_on_image_cv(image: np.ndarray, boxes: np.ndarray):
    """Plots bounding boxes on top of the image using OpenCV and saves the image to a file if save_path is provided.

    Args:
        image (np.ndarray): The image array.
        boxes (torch.Tensor): Tensor containing bounding boxes and other information with shape (num_boxes, 6).
        save_path (Path, optional): The file path to save the image with bounding boxes.
            If None, the image will not be saved.
    """
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        label = f"Conf: {conf:.2f}, Class: {cls:.0f}"

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        # Put the label above the bounding box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top_left = (x1, y1 - label_size[1] if y1 - label_size[1] > 0 else y1 + label_size[1])
        bottom_right = (x1 + label_size[0], y1)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)
        cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def recover_bounding_boxes(data: str, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Recovers bounding boxes from normalized format to xyxy format.

    Args:
        data (str): A string containing bounding box information in the format (class cx cy w h).
        image_shape (Tuple[int, int, int]): The shape of the image as (height, width, channels).

    Returns:
        np.ndarray: An array of bounding boxes in xyxy format.
    """
    lines = data.strip().split("\n")
    height, width = image_shape[:2]

    boxes = []
    for line in lines:
        class_id, cx, cy, w, h = map(float, line.split())
        x_center = cx * width
        y_center = cy * height

        box_width = w * width
        box_height = h * height

        x1 = x_center - (box_width / 2)
        y1 = y_center - (box_height / 2)
        x2 = x_center + (box_width / 2)
        y2 = y_center + (box_height / 2)

        boxes.append([x1, y1, x2 - x1, y2 - y1, class_id])

    return np.array(boxes)
