from typing import List, Union

import cv2
import numpy as np

from common.yolo.facial_orientation import FacialOrientation2D
from common.yolo.schema_loader import SchemaLoader
from common.yolo.yolo_results import YoloPose, Yolo, YoloPoseSorted, YoloSorted


class Drawer:
    """绘图工具类，负责在图像上绘制骨骼、关键点和边界框。"""

    def __init__(self, schema_loader: SchemaLoader):
        self.kpt_color_map = schema_loader.kpt_color_map
        self.skeleton_map = schema_loader.skeleton_map
        self.bbox_colors = schema_loader.bbox_colors


    def draw_skeletons_with_bboxes(self, image: np.ndarray, results: List[Union[YoloPose, YoloPoseSorted]],
                                   different_bbox: bool = False, show_pts: bool = True,
                                   show_names: bool = True) -> np.ndarray:
        """
        在图像上绘制骨骼和边界框。

        :param image: cv2 图像
        :param results: YoloPose 或 YoloPoseSorted 对象列表
        :param different_bbox: 是否使用不同颜色的边界框
        :param show_pts: 是否显示关键点
        :param show_names: 是否显示关键点名称
        :return: 绘制后的图像
        """
        for idx, pose in enumerate(results):
            # 绘制关键点
            if show_pts:
                for i, pt in enumerate(pose.pts):
                    if pt.conf > 0.2 and i in self.kpt_color_map:
                        kp = self.kpt_color_map[i]
                        cv2.circle(image, (pt.x, pt.y), 3, kp.color, -1)

                        if show_names:
                            cv2.putText(image, kp.name, (pt.x, pt.y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, kp.color, 1)

            # 绘制骨骼
            for bone in self.skeleton_map:
                srt_kp = pose.pts[bone.srt_kpt_id]
                dst_kp = pose.pts[bone.dst_kpt_id]

                if all([
                    srt_kp.conf > 0.2,
                    dst_kp.conf > 0.2,
                    srt_kp.x > 0, srt_kp.y > 0,
                    dst_kp.x > 0, dst_kp.y > 0
                ]):
                    cv2.line(image, (srt_kp.x, srt_kp.y), (dst_kp.x, dst_kp.y), bone.color, 2)

            # 确定边界框颜色
            if different_bbox:
                box_color = self.bbox_colors[idx % len(self.bbox_colors)]
            else:
                box_color = (255, 0, 0)  # 默认蓝色

            # 绘制边界框
            lx, ly, rx, ry = pose.lx, pose.ly, pose.rx, pose.ry
            cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

            # 创建标签文本
            label = f"Person: {pose.conf:.2f}"

            # 获取文本大小
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

            # 绘制标签文本
            cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image


    def draw_skeletons_without_bboxes(self, image: np.ndarray, results: List[Union[YoloPose, YoloPoseSorted]],
                                      show_pts: bool = True, show_names: bool = True) -> np.ndarray:
        """
        在图像上绘制骨骼，不包括边界框。

        :param image: cv2 图像
        :param results: YoloPose 或 YoloPoseSorted 对象列表
        :param show_pts: 是否显示关键点
        :param show_names: 是否显示关键点名称
        :return: 绘制后的图像
        """
        for pose in results:
            # 绘制关键点
            if show_pts:
                for i, pt in enumerate(pose.pts):
                    if pt.conf > 0.2 and i in self.kpt_color_map:
                        kp = self.kpt_color_map[i]
                        cv2.circle(image, (pt.x, pt.y), 3, kp.color, -1)

                        if show_names:
                            cv2.putText(image, kp.name, (pt.x, pt.y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, kp.color, 1)

            # 绘制骨骼
            for bone in self.skeleton_map:
                srt_kp = pose.pts[bone.srt_kpt_id]
                dst_kp = pose.pts[bone.dst_kpt_id]

                if all([
                    srt_kp.conf > 0.2,
                    dst_kp.conf > 0.2,
                    srt_kp.x > 0, srt_kp.y > 0,
                    dst_kp.x > 0, dst_kp.y > 0
                ]):
                    cv2.line(image, (srt_kp.x, srt_kp.y), (dst_kp.x, dst_kp.y), bone.color, 2)

        return image


    def draw_bboxes_with_labels(self, image: np.ndarray, results: List[Union[Yolo, YoloSorted]],
                                labels: List[str]) -> np.ndarray:
        """
        在图像上绘制带标签的边界框。

        :param image: cv2 图像
        :param results: Yolo 或 YoloSorted 对象列表
        :param labels: 类别标签列表
        :return: 绘制后的图像
        """
        for yolo in results:
            lx, ly, rx, ry, cls, conf = yolo.lx, yolo.ly, yolo.rx, yolo.ry, yolo.cls, yolo.conf

            # 根据类别选择颜色
            box_color = self.bbox_colors[cls % len(self.bbox_colors)]

            # 绘制边界框
            cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

            # 创建标签文本
            label = f"{labels[cls]}: {conf:.2f}"

            # 获取文本大小
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制文本背景
            cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), box_color, cv2.FILLED)

            # 绘制标签文本
            cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image


    def draw_bboxes_without_labels(self, image: np.ndarray, results: List[Union[Yolo, YoloSorted]]) -> np.ndarray:
        """
        在图像上绘制不带标签的边界框。

        :param image: cv2 图像
        :param results: Yolo 或 YoloSorted 对象列表
        :return: 绘制后的图像
        """
        for yolo in results:
            lx, ly, rx, ry, cls, conf = yolo.lx, yolo.ly, yolo.rx, yolo.ry, yolo.cls, yolo.conf

            # 根据类别选择颜色
            box_color = self.bbox_colors[cls % len(self.bbox_colors)]

            # 绘制边界框
            cv2.rectangle(image, (lx, ly), (rx, ry), box_color, 2)

        return image


    def draw_facial_vectors_2d(self, frame: np.ndarray, orientation_vectors: List[FacialOrientation2D],
                               different_vectors: bool = False, show_legend: bool = False) -> np.ndarray:
        """
        在图像上绘制面部方向向量。

        :param frame: cv2 图像
        :param orientation_vectors: FacialOrientation2D 对象列表
        :param different_vectors: 是否使用不同颜色的向量
        :param show_legend: 是否显示图例
        :return: 绘制后的图像
        """
        for idx, vector in enumerate(orientation_vectors):
            # 确定向量颜色
            if different_vectors:
                vector_color = self.bbox_colors[idx % len(self.bbox_colors)]
            else:
                vector_color = (255, 0, 0)  # 默认蓝色

            # 绘制箭头
            cv2.arrowedLine(frame, (vector.origin_x, vector.origin_y), (vector.dest_x, vector.dest_y),
                            vector_color, 2)

            # 绘制图例
            if show_legend:
                face_direction = str(vector)
                cv2.putText(frame, face_direction, (vector.origin_x + 5, vector.origin_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, vector_color, 1)

        return frame
