import numpy as np
from typing import List, Tuple

from common.yolo.yolo_results import Yolo, YoloPose, YoloPoint, YoloPoseSorted, YoloSorted


###### Convert yolo.results to a list of Yolo/YoloPose objects ######
def results_to_yolo_list(results) -> List[Yolo]:
    # A list to store the results
    yolo_objects = []

    # If the results is not a list
    if isinstance(results, list):
        raise ValueError("The results object is a list.")

    # If boxes is not in the results
    if not hasattr(results, 'boxes'):
        raise ValueError("The results object does not have the attribute 'boxes'.")

    # Loop through the results
    for det in results.boxes:
        # Get the bounding box
        x1, y1, x2, y2 = map(int, det.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = det.conf.cpu().numpy().item()

        # Get the class
        cls = int(det.cls[0])

        # Create a yolo object to store the results
        yolo = Yolo(x1, y1, x2, y2, cls, conf)

        # Append the yolo object to the list
        yolo_objects.append(yolo)

    return yolo_objects


def results_to_pose_list(results) -> List[YoloPose]:
    # A list to store the results
    yolo_objects = []

    # If the results is a list
    if isinstance(results, list):
        raise ValueError("The results object is a list.")

    # If boxes is not in the results
    if not hasattr(results, 'boxes'):
        raise ValueError("The results object does not have the attribute 'boxes'.")

    # If keypoints is not in the results
    if not hasattr(results, 'keypoints'):
        raise ValueError("The results object does not have the attribute 'keypoints'.")

    # Loop through the results
    for box, kpts in zip(results.boxes, results.keypoints):
        # Get the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get the confidence
        # conf = det.conf # CUDA tensor
        conf = box.conf.cpu().numpy().item()

        # Create a list to store the keypoints
        keypoints = []

        # Squeeze the keypoints
        kpts_xy = kpts.xy.cpu().numpy().squeeze()  # (17, 2)
        kpts_conf = kpts.conf.cpu().numpy().squeeze()  # (17,)

        # Loop through the keypoints
        for (x, y), c in zip(kpts_xy, kpts_conf):
            # Create a yolo point object
            point = YoloPoint(int(x), int(y), c)

            # Append the yolo point object to the list
            keypoints.append(point)

        # Create a yolo pose object to store the results
        pose = YoloPose(x1, y1, x2, y2, 0, conf, keypoints)

        # Append the yolo pose object to the list
        yolo_objects.append(pose)

    return yolo_objects


###### list of YoloPoint ######
def yolo_point_list_to_numpy(data: list[YoloPoint]) -> np.ndarray:
    """
    Converts a list of YoloPoint objects to a numpy array.

    :param data: list of YoloPoint objects
    :return: numpy array
    """
    numpy_list = []
    for d in data:
        numpy_list.append([d.x, d.y, d.conf])
    
    return np.array(numpy_list)


def numpy_to_yolo_point_list(data: np.ndarray) -> List[YoloPoint]:
    """
    Converts a numpy array to a list of YoloPoint objects.

    :param data: numpy array
    :return: list of YoloPoint objects
    """
    output = []
    for row in data:
        y = YoloPoint()
        y.from_list(row.tolist())
        output.append(y)

    return output


###### Convert a list of Yolo/YoloPose objects to a numpy array ######
def yolo_list_to_numpy(data: list[Yolo]) -> np.ndarray:
    """
    Converts a list of Yolo objects to a numpy array with reordered attributes.
    The order will be: [x1, y1, x2, y2, conf, cls]
    """
    numpy_list = []
    for d in data:
        # Order attributes as [x1, y1, x2, y2, conf, cls]
        numpy_list.append([d.lx, d.ly, d.rx, d.ry, d.conf, d.cls])
    
    return np.array(numpy_list)


def pose_list_to_numpy(data: list[YoloPose]) -> np.ndarray:
    """
    Converts a list of YoloPose objects to a fully flattened numpy array with reordered attributes.
    Each row will contain [x1, y1, x2, y2, conf, cls, ...] followed by the flattened keypoints.
    """
    numpy_list = []
    for d in data:
        # Flatten keypoints as a flat list for each person
        pts_flat = [value for pt in d.pts for value in pt.to_list()]
        # Combine attributes in the desired order: [x1, y1, x2, y2, conf, cls]
        pose_array = np.array([d.lx, d.ly, d.rx, d.ry, d.conf, d.cls] + pts_flat)
        numpy_list.append(pose_array)

    return np.array(numpy_list)  # Shape: (num_poses, 6 + 17*3)


###### Convert a numpy array to a list of Yolo/YoloPose objects ######

def numpy_to_yolo_list(data: np.ndarray) -> List[Yolo]:
    """
    Converts a numpy array to a list of Yolo objects with the updated attribute order.
    Each row in `data` should contain the main attributes in the order: [x1, y1, x2, y2, conf, cls].
    """
    yolo_list = []
    for row in data:
        # Extract attributes in the order [x1, y1, x2, y2, conf, cls]
        lx, ly, rx, ry, conf, cls = row[:6]
        yolo = Yolo(lx=int(lx), ly=int(ly), rx=int(rx), ry=int(ry), cls=int(cls), conf=float(conf))
        yolo_list.append(yolo)
    return yolo_list


def numpy_to_pose_list(data: np.ndarray) -> List[YoloPose]:
    """
    Converts a fully flattened numpy array back to a list of YoloPose objects.
    Each row in `data` should contain the main attributes followed by the flattened
    keypoints for a single detected pose.
    """
    pose_list = []
    for row in data:
        # Extract main attributes
        lx, ly, rx, ry, conf, cls, = row[:6]

        # Extract and reshape keypoints data for each pose
        pts_data = row[6:].reshape(-1, 3)  # Reshape to (17, 3)
        pts = [YoloPoint(x=int(pt[0]), y=int(pt[1]), conf=float(pt[2])) for pt in pts_data]

        # Create YoloPose object
        yolo_pose = YoloPose(lx=int(lx), ly=int(ly), rx=int(rx), ry=int(ry), cls=int(cls), conf=float(conf), pts=pts)
        pose_list.append(yolo_pose)
    
    return pose_list


###### Sort the results and tracker IDs to separate lists ######

# def numpy_to_sorted_pose_list(sorted_data: np.ndarray) -> List[YoloPoseSorted]:
#     """
#     将排序后的数据转换为 YoloPoseSorted 对象的列表。

#     :param sorted_data: 排序后的 numpy 数组，第一列为 tracker ID，其余列为 pose 数据
#     :return: List[YoloPoseSorted]
#     """
#     if sorted_data.ndim != 2:
#         raise ValueError("sorted_data must be a 2D numpy array")

#     # The first column contains the tracker ID
#     tracker_ids = sorted_data[:, 0].astype(int).tolist()

#     # The rest of the columns contain the pose data
#     pose_data = sorted_data[:, 1:]

#     # Convert the pose data to a list of YoloPose objects
#     pose_list = numpy_to_pose_list(pose_data)

#     # Combine tracker_ids with pose_list to create YoloPoseSorted objects
#     # pose_sorted_list = [YoloPoseSorted(oid=tid, **pose.to_dict()) for tid, pose in zip(tracker_ids, pose_list)]
#     pose_sorted_list = []
#     for tid, pose in enumerate(tracker_ids, pose_list):
#         sorted = YoloPoseSorted(oid=tid, lx=pose.lx, ly=pose.ly, rx=pose.rx, ry=pose.ry,
#                                        conf=pose.conf, cls=pose.cls, pts=pose.pts)
#         pose_sorted_list.append(sorted)

#     # pose_sorted_list = [YoloPoseSorted(oid=tid, lx=pose.lx, ly=pose.ly, rx=pose.rx, ry=pose.ry,
#     #                                    conf=pose.conf, cls=pose.cls, pts=pose.pts)
#     #                     for tid, pose in enumerate(tracker_ids, pose_list)]

#     return pose_sorted_list

def numpy_to_sorted_pose_list(sorted_data: np.ndarray) -> List[YoloPoseSorted]:
    """
    将排序后的数据转换为 YoloPoseSorted 对象的列表。

    :param sorted_data: 排序后的 numpy 数组，第一列为 tracker ID，其余列为 pose 数据
    :return: List[YoloPoseSorted]
    """
    if sorted_data.ndim != 2:
        raise ValueError("sorted_data must be a 2D numpy array")

    # The first column contains the tracker ID
    tracker_ids = sorted_data[:, 0].astype(int).tolist()

    # The rest of the columns contain the pose data
    pose_data = sorted_data[:, 1:]

    # Convert the pose data to a list of YoloPose objects
    pose_list = numpy_to_pose_list(pose_data)

    # Combine tracker_ids with pose_list to create YoloPoseSorted objects
    pose_sorted_list = []
    for tid, pose in zip(tracker_ids, pose_list):
        sorted_pose = YoloPoseSorted(oid=tid, lx=pose.lx, ly=pose.ly, rx=pose.rx, ry=pose.ry,
                                     conf=pose.conf, cls=pose.cls, pts=pose.pts)
        pose_sorted_list.append(sorted_pose)

    return pose_sorted_list



def numpy_to_sorted_yolo_list(sorted_data: np.ndarray) -> List[YoloSorted]:
    """
    将排序后的数据转换为 YoloSorted 对象的列表。

    :param sorted_data: 排序后的 numpy 数组，第一列为 tracker ID，其余列为 yolo 数据
    :return: List[YoloSorted]
    """
    if sorted_data.ndim != 2:
        raise ValueError("sorted_data must be a 2D numpy array")

    # The first column contains the tracker ID
    tracker_ids = sorted_data[:, 0].astype(int).tolist()

    # The rest of the columns contain the yolo data
    yolo_data = sorted_data[:, 1:]

    # Convert the yolo data to a list of Yolo objects
    yolo_list = numpy_to_yolo_list(yolo_data)

    # Combine tracker_ids with yolo_list to create YoloSorted objects
    yolo_sorted_list = [YoloSorted(oid=tid, **yolo.to_dict()) for tid, yolo in zip(tracker_ids, yolo_list)]

    return yolo_sorted_list
