from common.yolo.ultralytics_results_wrapper import YoloPose
from typing import List


class FacialOrientation2D:
    
    def __init__(self, length=100):
        """
        Initialize the FacialOrientation2D class with default values.

        :param length: Length of the direction vector for visualization purposes.
        """
        # Origin coordinates
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Direction vector of face orientation
        self.vec_x = 0.0
        self.vec_y = 0.0

        # End point of the face direction vector
        self.dest_x = 0.0
        self.dest_y = 0.0

        # Top-left corner of the bounding box
        self.lx = 0
        self.ly = 0

        # Vector length
        self.length = length

        # Face orientation state: -1 (unknown), 0 (front), 1 (left), 2 (right) and 3 (back)
        self.orientation = -1

    def check_orientation(self, pose: YoloPose):
        """
        Determine facial orientation based on keypoints in the pose.

        :param pose: YoloPose object with keypoints for face orientation analysis.
        """

        # Update the left-top corner of the bounding box
        self.lx, self.ly = pose.lx, pose.ly

        # Get the keypoints for nose, right ear, and left ear
        nose_pt = pose.pts[0]
        right_ear_pt = pose.pts[3]
        left_ear_pt = pose.pts[4]

        # Temporary variables for ear points
        ear_tmp_x, ear_tmp_y = 0, 0

        # Front, right, and left face orientations
        if self._is_valid_pt(nose_pt):

            # Front face orientation
            if self._is_valid_pt(right_ear_pt) and self._is_valid_pt(left_ear_pt):
                self.orientation = 0 

            # Right orientation
            elif self._is_valid_pt(right_ear_pt) and not self._is_valid_pt(left_ear_pt):
                self.orientation = 2 
                ear_tmp_x, ear_tmp_y = right_ear_pt.x, right_ear_pt.y
            
            # Left orientation
            elif self._is_valid_pt(left_ear_pt) and not self._is_valid_pt(right_ear_pt):
                self.orientation = 1 
                ear_tmp_x, ear_tmp_y = left_ear_pt.x, left_ear_pt.y
            
            # Unrecognized orientation
            else:
                self.orientation = -1  # Unrecognized

        # Back orientation
        elif not self._is_valid_pt(nose_pt):

            # Back orientation
            if self._is_valid_pt(right_ear_pt) or self._is_valid_pt(left_ear_pt):
                self.orientation = 3

            # Unrecognized orientation
            else:
                self.orientation = -1
        
        # Unrecognized orientation
        else:
            self.orientation = -1 

        # Calculate the face orientation vector based on the keypoints
        if self.orientation in [1, 2]:
            self._calculate_vector(nose_pt, ear_tmp_x, ear_tmp_y)

        # Do not calculate vector for front-facing orientation
        elif self.orientation == 3:
            self._set_front_face(nose_pt)

        # Unrecognized orientation
        else:
            self._set_unrecognized(pose)

        # Convert to integer coordinates for consistency
        self.origin_x, self.origin_y = int(self.origin_x), int(self.origin_y)
        self.dest_x, self.dest_y = int(self.dest_x), int(self.dest_y)

    def _calculate_vector(self, nose_pt, ear_x, ear_y):
        """Calculate face orientation vector when the face is turned sideways."""
        self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
        self.vec_x, self.vec_y = nose_pt.x - ear_x, nose_pt.y - ear_y
        self.vec_x, self.vec_y = self._normalize(self.vec_x, self.vec_y)
        self.dest_x = self.origin_x + self.vec_x * self.length
        self.dest_y = self.origin_y + self.vec_y * self.length

    def _set_front_face(self, nose_pt):
        """Set values for a front-facing orientation."""
        self.origin_x, self.origin_y = nose_pt.x, nose_pt.y
        self.vec_x = self.vec_y = 0
        self.dest_x, self.dest_y = nose_pt.x, nose_pt.y

    def _set_unrecognized(self, pose):
        """Set values for unrecognized orientation."""
        self.origin_x = self.origin_y =  0
        self.vec_x = self.vec_y = 0
        self.dest_x = self.dest_y = 0

    @staticmethod
    def _is_valid_pt(pt):
        """Check if a keypoint is valid based on confidence and coordinates."""
        return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

    @staticmethod
    def _normalize(x, y):
        """Normalize a vector to unit length."""
        vec_len = (x ** 2 + y ** 2) ** 0.5
        return x / vec_len, y / vec_len

    def __str__(self):
        """String representation based on face orientation."""
        orientation_texts = { 0: "Face front", 1: "Face left", 2: "Face right", 3: "Back", -1: "Unknown"}
        return orientation_texts.get(self.orientation, "Unknown")


def detect_facial_vectors(results: List[YoloPose]) -> List[FacialOrientation2D]:
    """
    Process a list of YoloPose objects to determine facial orientation vectors.

    :param results: List of YoloPose objects with keypoints for face orientation.
    :return: List of FacialOrientation2D objects with calculated orientations.
    """
    facial_vectors = []
    for pose in results:
        facial_vector = FacialOrientation2D()
        facial_vector.check_orientation(pose)
        facial_vectors.append(facial_vector)
    return facial_vectors


def detect_facial_orientations(results: List[YoloPose], return_type="str") -> List:

    # 顔の向きvectorを検出する
    facial_vectors = detect_facial_vectors(results)

    facial_orientations = []
    for vector in facial_vectors:
   
        # 顔の向きを文字列で返す
        if return_type == "str":
            orientation = vector.orientation

            if orientation == 0:
                facial_orientations.append("Face front")
            if orientation == 1:
                facial_orientations.append("Face left")
            elif orientation == 2:
                facial_orientations.append("Face right")
            elif orientation == 3:
                facial_orientations.append("Face back")
            else:
                facial_orientations.append("Unknown")

        # 顔の向きを数値で返す
        else:
            facial_orientations.append(vector.orientation)

    return facial_orientations
