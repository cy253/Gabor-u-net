import SimpleITK as sitk
import numpy as np


def calculate_hd95(prediction, ground_truth):
    # 将预测和真实分割结果加载为 SimpleITK 图像
    pred_image = sitk.ReadImage(prediction, sitk.sitkUInt8)
    gt_image = sitk.ReadImage(ground_truth, sitk.sitkUInt8)

    # 将 SimpleITK 图像转换为 NumPy 数组
    pred_array = sitk.GetArrayFromImage(pred_image)
    gt_array = sitk.GetArrayFromImage(gt_image)

    # 计算 HD95
    hd95 = compute_hd95(pred_array, gt_array)

    return hd95


def compute_hd95(pred_array, gt_array):
    # 将布尔类型的数组转换为整数类型
    pred_array = pred_array.astype(np.uint8)
    gt_array = gt_array.astype(np.uint8)

    # 计算距离图
    distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(sitk.GetImageFromArray(pred_array == 0)))

    # 获取预测和真实分割结果的非零像素坐标
    pred_coords = np.argwhere(pred_array != 0)
    gt_coords = np.argwhere(gt_array != 0)

    # 计算每个真实分割像素到最近的预测分割像素的距离
    distances = []
    for gt_coord in gt_coords:
        gt_point = [float(coord) for coord in gt_coord]
        distance = distance_map.GetPixel(tuple(gt_point))
        distances.append(distance)

    # 将距离按升序排列
    distances.sort()

    # 计算 HD95
    hd95_index = int(0.95 * len(distances))
    hd95 = distances[hd95_index]

    return hd95

# 用法示例
prediction_path = "E:\\code\\shenduxuexi\\brats-unet-main\\preditions\\unet\\00_pred.nii.gz"
ground_truth_path = "E:\\code\\shenduxuexi\\brats-unet-main\\preditions\\unet\\00_gt.nii.gz"

hd95_value = calculate_hd95(prediction_path, ground_truth_path)
print("HD95 value:", hd95_value)
