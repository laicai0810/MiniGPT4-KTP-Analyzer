import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
from tqdm import tqdm
import os
import argparse
import random


# --- 图像处理辅助函数 (已修正和优化) ---

def find_largest_quadrilateral(image):
    """在图像中找到最大的四边形轮廓。"""
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def order_points(pts):
    """对四边形的四个点进行排序：上左, 上右, 下右, 下左。"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(image, pts):
    """对图像进行透视变换。"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def correct_orientation_opencv(image_cv):
    """使用OpenCV的Haar分类器校正图像方向。"""
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found. Expected at '{cascade_path}'")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    pil_img = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    for angle in [0, 270, 180, 90]:
        rotated_pil = pil_img.rotate(angle, expand=True)
        rotated_cv = cv2.cvtColor(np.array(rotated_pil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(rotated_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return rotated_pil
    return pil_img


# --- JSON到对话转换辅助函数 (性别专精版) ---

def create_conversational_data(json_metadata_dict):
    """
    根据JSON元数据创建只关于“性别”的对话。
    """
    try:
        # 只从JSON字典中提取性别信息
        gender = json_metadata_dict.get('gender', '未知')

        # 将印尼语的性别翻译为中文，以提高模型训练效果
        if gender.upper() == 'PEREMPUAN':
            gender_chinese = '女性'
        elif gender.upper() == 'LAKI-LAKI':  # 假设男性的值是 LAKI-LAKI
            gender_chinese = '男性'
        else:
            gender_chinese = '未知'

        # 定义一组固定的、关于性别的问题模板，以增加数据多样性
        question_templates = [
            "###Human: 这张身份证上的性别是什么？",
            "###Human: 请告诉我这个人的性别。",
            "###Human: 这张图片里的人是男性还是女性？",
            "###Human: 识别一下图中的性别信息。"
        ]

        # 随机选择一个问题模板
        human_question = random.choice(question_templates)

        # 构建固定的回答
        assistant_answer = f"###Assistant: 根据证件信息，性别为{gender_chinese}。"

        return f"{human_question} {assistant_answer}"

    except Exception as e:
        return "###Human: 请描述这张图片中的性别。 ###Assistant: 无法从元数据中提取性别信息。"


def process_csv_and_generate_data(csv_path, image_base_dir, image_column, json_column, output_image_dir,
                                  output_jsonl_path):
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件未找到于 {csv_path}")
        return

    try:
        # --- 错误修复 ---
        # 根据调试输出，CSV文件应使用逗号作为分隔符，而不是制表符。
        df = pd.read_csv(csv_path, sep=',')
        # 移除列名中可能存在的前后空格，这是一个常见问题。
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"错误: 读取CSV文件失败: {e}")
        return

    if image_column not in df.columns or json_column not in df.columns:
        print(f"错误: CSV中未找到指定的图像列 '{image_column}' 或JSON列 '{json_column}'。")
        # 增加调试信息：打印出所有pandas识别到的列名
        print("可用的列名:", df.columns.tolist())
        return

    os.makedirs(output_image_dir, exist_ok=True)
    processed_data_for_jsonl = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="正在处理数据"):
        original_image_path_relative = row[image_column]
        json_metadata_str = row[json_column]

        if not isinstance(original_image_path_relative, str):
            continue

        normalized_path = original_image_path_relative.replace("\\", "/").lstrip("./")
        full_image_path = os.path.join(image_base_dir, normalized_path)

        if not os.path.exists(full_image_path):
            continue

        base_filename = os.path.basename(full_image_path)
        processed_image_filename = f"processed_{base_filename}"
        processed_image_path = os.path.join(output_image_dir, processed_image_filename)

        try:
            original_cv_image = cv2.imread(full_image_path)
            if original_cv_image is None:
                continue

            quad_pts = find_largest_quadrilateral(original_cv_image)
            cropped_image_cv = perspective_transform(original_cv_image,
                                                     quad_pts) if quad_pts is not None else original_cv_image

            oriented_pil_image = correct_orientation_opencv(cropped_image_cv)
            oriented_pil_image.save(processed_image_path)

            metadata_dict = json.loads(json_metadata_str) if isinstance(json_metadata_str, str) else {}
            conversational_string = create_conversational_data(metadata_dict)

            processed_data_for_jsonl.append({
                "image": processed_image_path.replace("\\", "/"),
                "value": conversational_string
            })

        except Exception as e:
            continue

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in processed_data_for_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"成功生成JSONL文件于 {output_jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为MiniGPT-4微调处理来自CSV的图像和JSON元数据。")
    parser.add_argument("--csv_path", type=str, required=True, help="输入CSV文件的路径。")
    parser.add_argument("--image_base_dir", type=str, default="", help="用于解析CSV中相对图像路径的根目录。")
    parser.add_argument("--image_column", type=str, default="a001_证件照片本地路径", help="CSV中包含图像路径的列名。")
    parser.add_argument("--json_column", type=str, default="a001_txt_报文内容", help="CSV中包含JSON元数据字符串的列名。")
    parser.add_argument("--output_image_dir", type=str, default="processed_images_gender", help="保存处理后图像的目录。")
    parser.add_argument("--output_jsonl_path", type=str, default="finetune_data_gender.jsonl",
                        help="保存输出JSONL文件的路径。")

    args = parser.parse_args()

    process_csv_and_generate_data(
        args.csv_path,
        args.image_base_dir,
        args.image_column,
        args.json_column,
        args.output_image_dir,
        args.output_jsonl_path
    )