# verify_ktp_data.py
import pandas as pd
import json
import re
from datetime import datetime
import time
import os
from PIL import Image, ImageOps, UnidentifiedImageError  # 增加 UnidentifiedImageError
import argparse
import torch
from tqdm import tqdm

# --- MiniGPT-4 相关导入 ---
try:
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import CONV_VISION_Vicuna0, CONV_VISION_LLama2, Chat

    print("MiniGPT-4 相关库导入成功。")
except ImportError as e:
    print(f"错误: 导入MiniGPT-4库失败: {e}。请确保环境配置正确。")
    exit()

# --- KTP Analyzer 导入 ---
try:
    from ktp_analyzer import straighten_and_enhance_image

    print("从 'ktp_analyzer.py' 成功导入 straighten_and_enhance_image 函数。")
    KTP_ANALYZER_AVAILABLE = True
except ImportError:
    print("警告: 未找到或无法导入 ktp_analyzer.py。KTP图像的透视校正功能将不可用。")
    KTP_ANALYZER_AVAILABLE = False


    def straighten_and_enhance_image(image_pil):  # 存根函数
        # print("ktp_analyzer.straighten_and_enhance_image 未加载，返回原始图像。")
        if image_pil.mode != 'RGB':
            return image_pil.convert('RGB')
        return image_pil.copy()

# 全局变量用于MiniGPT-4模型
mini_gpt4_model = None
mini_gpt4_vis_processor = None
mini_gpt4_chat_instance = None
MINIGPT4_CONV_VISION = None


# --- MiniGPT-4 模型初始化函数 ---
def initialize_minigpt4(cfg_path="eval_configs/minigpt4_eval.yaml", gpu_id=0):
    global mini_gpt4_model, mini_gpt4_vis_processor, mini_gpt4_chat_instance, MINIGPT4_CONV_VISION
    if mini_gpt4_chat_instance is not None:
        print("MiniGPT-4 模型已初始化。")
        return True

    print("正在初始化 MiniGPT-4 模型...")
    try:
        parser = argparse.ArgumentParser(description="MiniGPT-4 KTP Verification")
        parser.add_argument("--cfg-path", default=cfg_path, help="配置文件路径。")
        parser.add_argument("--gpu-id", type=int, default=gpu_id, help="指定GPU ID。")
        parser.add_argument("--options", nargs="+", help="覆盖配置文件中的某些设置。")

        # 模拟命令行参数，因为我们是从脚本内部调用
        args_list = ['--cfg-path', cfg_path, '--gpu-id', str(gpu_id)]
        # 使用 parse_known_args 以允许未定义的其他参数（如果存在于配置文件中但此处未定义）
        args, _ = parser.parse_known_args(args_list)

        cfg = Config(args)
        # 支持的对话模板
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
        model_config = cfg.model_cfg

        # 根据配置文件选择对话模板
        if model_config.model_type in conv_dict:
            MINIGPT4_CONV_VISION = conv_dict[model_config.model_type]
        else:
            print(f"警告: 未知的 model_type '{model_config.model_type}'. 将使用默认对话模板 Vicuna0。")
            MINIGPT4_CONV_VISION = CONV_VISION_Vicuna0  # 默认

        model_config.device_8bit = args.gpu_id  # 设置设备
        model_cls = registry.get_model_class(model_config.arch)

        # 暂时降低 transformers 日志级别，避免加载模型时过多信息
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        mini_gpt4_model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
        hf_logging.set_verbosity_warning()  # 恢复日志级别

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        # 确保 vis_processor_cfg 包含 'name' 属性
        if hasattr(vis_processor_cfg, 'name'):
            mini_gpt4_vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
                vis_processor_cfg)
        else:
            # 如果配置中缺少 'name'，则抛出错误或设置一个默认值（如果适用）
            raise ValueError("视觉处理器配置中缺少 'name' 属性。请检查您的配置文件。")

        mini_gpt4_chat_instance = Chat(mini_gpt4_model, mini_gpt4_vis_processor, device=f'cuda:{args.gpu_id}')
        print("MiniGPT-4 模型初始化完成。")
        return True
    except Exception as e:
        print(f"MiniGPT-4 模型初始化失败: {e}")
        # print(traceback.format_exc()) # 打印详细堆栈信息以供调试
        return False


# --- MiniGPT-4 内部调用函数 (避免重复代码) ---
def _call_minigpt4_for_gender(pil_image_input):
    global mini_gpt4_chat_instance, MINIGPT4_CONV_VISION
    # 此函数假设 pil_image_input 已经是处理好的 RGB PIL Image

    temp_chat_state = MINIGPT4_CONV_VISION.copy()
    current_img_list = []

    # 1. upload_img: 将 PIL Image 添加到 current_img_list, 并向对话状态添加占位符
    mini_gpt4_chat_instance.upload_img(pil_image_input, temp_chat_state, current_img_list)

    # 校验 upload_img 后：current_img_list[0] 是否为传入的 PIL Image
    if not current_img_list or not isinstance(current_img_list[0], Image.Image) or current_img_list[
        0] != pil_image_input:
        return "LLM图像上传失败 (图像未添加到处理列表)"

        # 2. encode_img: 处理 current_img_list 中的 PIL Image,
    #    并将其替换为 Tensor 嵌入。
    try:
        mini_gpt4_chat_instance.encode_img(current_img_list)
    except Exception as e:
        # print(f"调试: encode_img 过程中发生错误: {e}") # 可选的调试打印
        return f"LLM图像编码处理失败: {str(e)[:100]}"

        # 校验 encode_img 后：current_img_list[0] 是否为 torch.Tensor
    if not current_img_list or not isinstance(current_img_list[0], torch.Tensor):
        return "LLM图像转换或编码失败 (未生成有效图像张量)"

    prompt = "Analyze the person in the photograph. Is their apparent gender male or female? Please respond with only the single word 'male' or 'female'."

    # 3. ask: 将文本提示添加到对话状态。
    #    conversation.py 中的 ask() 方法会处理与图像占位符消息的合并（如果它是最后一条消息）。
    mini_gpt4_chat_instance.ask(prompt, temp_chat_state)

    # 4. answer: 模型生成回复。
    #    img_list (现在包含 Tensor 嵌入) 被传递给 model.get_context_emb 使用。
    try:
        llm_response = mini_gpt4_chat_instance.answer(
            conv=temp_chat_state, img_list=current_img_list,
            max_new_tokens=10, num_beams=1, temperature=0.01  # 使用低温以获得更确定的输出
        )[0]
    except Exception as e:
        # print(f"调试: answer 生成过程中发生错误: {e}") # 可选的调试打印
        return f"LLM回答生成失败: {str(e)[:100]}"

    response_lower = llm_response.lower().strip()
    if any(keyword in response_lower for keyword in ['male', 'man', 'laki', 'pria']):
        return "LAKI-LAKI"
    elif any(keyword in response_lower for keyword in ['female', 'woman', 'perempuan', 'wanita']):
        return "PEREMPUAN"
    else:
        # print(f"调试: LLM 性别回复不明确: '{llm_response}'") # 可选的调试打印
        return "无法确定_LLM"


# --- MiniGPT-4 视觉性别判断函数 (带预处理重试逻辑) ---
def get_visual_gender_with_minigpt4(image_path):
    global mini_gpt4_chat_instance, MINIGPT4_CONV_VISION, KTP_ANALYZER_AVAILABLE

    if mini_gpt4_chat_instance is None or MINIGPT4_CONV_VISION is None:
        return "模型或对话模板未初始化", 0.0, "原始图像"

    start_time_llm_total = time.time()
    final_gender_result = "处理失败_LLM"
    best_attempt_description = "原始图像"

    try:
        original_pil_image = Image.open(image_path)
        original_pil_image = ImageOps.exif_transpose(original_pil_image)  # 处理EXIF旋转
        if original_pil_image.mode != 'RGB':  # 确保是RGB格式
            original_pil_image = original_pil_image.convert('RGB')
    except FileNotFoundError:
        return "图片未找到", 0.0, best_attempt_description
    except UnidentifiedImageError:  # Pillow无法识别图像格式
        return "无法识别的图像格式", 0.0, best_attempt_description
    except Exception as e:  # 其他可能的图片加载错误
        return f"图片加载错误: {str(e)[:50]}", 0.0, best_attempt_description

    # 首次尝试：直接使用原始（已EXIF校正和RGB转换的）图像
    # print(f"调试: 首次尝试LLM性别判断，图像路径: {image_path}")
    current_gender_try = _call_minigpt4_for_gender(original_pil_image.copy())  # 使用副本以防后续处理修改原图
    # print(f"调试: 首次尝试结果: {current_gender_try}")

    if current_gender_try in ["LAKI-LAKI", "PEREMPUAN"]:
        final_gender_result = current_gender_try
    else:  # 如果首次尝试不确定或失败，则进行预处理和旋转尝试
        # print(f"调试: 首次尝试不确定 ({current_gender_try})，尝试预处理...")
        processing_attempts = [("原始图像+透视校正", 0)]  # (描述, 旋转角度)
        if KTP_ANALYZER_AVAILABLE:  # 只有在ktp_analyzer可用时才添加旋转
            processing_attempts.extend([
                ("旋转90度+透视校正", 90),
                ("旋转-90度+透视校正", -90),
                ("旋转180度+透视校正", 180)
            ])
        # 如果ktp_analyzer不可用，processing_attempts 只包含 ("原始图像+透视校正", 0)

        for desc, angle in processing_attempts:
            # print(f"调试: 尝试预处理: {desc}")
            image_to_process = original_pil_image.copy()  # 每次都从原始副本开始
            if angle != 0:
                image_to_process = image_to_process.rotate(angle, expand=True)  # expand=True 避免旋转时图像被裁剪

            # 应用 straighten_and_enhance_image (透视校正或存根函数)
            # straighten_and_enhance_image 应该返回 RGB PIL Image
            try:
                corrected_image = straighten_and_enhance_image(image_to_process)
                if corrected_image.mode != 'RGB':  # 再次确保是RGB
                    corrected_image = corrected_image.convert('RGB')
            except Exception as e_corr:
                print(f"警告: 应用透视校正/旋转 ({desc}) 时出错: {e_corr}。跳过此尝试。")
                continue

            # print(f"调试: 使用处理后图像 ({desc}) 进行LLM性别判断...")
            current_gender_try_processed = _call_minigpt4_for_gender(
                corrected_image)  # 这里也用 corrected_image 的副本，虽然_call_minigpt4_for_gender内部也会copy
            # print(f"调试: {desc} 结果: {current_gender_try_processed}")

            if current_gender_try_processed in ["LAKI-LAKI", "PEREMPUAN"]:
                final_gender_result = current_gender_try_processed
                best_attempt_description = desc
                break  # 找到明确性别，停止尝试
            elif final_gender_result == "处理失败_LLM" or final_gender_result == "无法确定_LLM" or final_gender_result.startswith(
                    "LLM图像") or final_gender_result.startswith("LLM回答"):
                # 如果之前的尝试也是失败或不确定，更新为当前的（可能仍然不确定的或失败的）结果
                # 这样至少我们有一个来自处理后图像的结果，即使它仍然不确定或指示了新的错误
                final_gender_result = current_gender_try_processed
                best_attempt_description = desc  # 更新描述，即使结果仍不确定

    end_time_llm_total = time.time()
    llm_total_processing_time = round(end_time_llm_total - start_time_llm_total, 3)
    return final_gender_result, llm_total_processing_time, best_attempt_description


# --- 辅助函数 ---
def normalize_gender_from_text(gender_text):
    if not isinstance(gender_text, str): return "无法识别_文本"
    gt = gender_text.upper()
    if "LAKI" in gt:
        return "LAKI-LAKI"
    elif "PEREMPUAN" in gt:
        return "PEREMPUAN"
    return "无法识别_文本"


def get_gender_from_nik(nik):
    if not isinstance(nik, str) or len(nik) != 16 or not nik.isdigit(): return "NIK无效"
    try:
        day_digits = int(nik[6:8])
        if 1 <= day_digits <= 31:
            return "LAKI-LAKI"
        elif 41 <= day_digits <= 71:
            return "PEREMPUAN"
        else:
            return "NIK日期码无效"
    except ValueError:  # nik[6:8] 不是数字
        return "NIK无效"


def get_birthdate_from_nik(nik):
    if not isinstance(nik, str) or len(nik) != 16 or not nik.isdigit(): return None
    try:
        day_digits = int(nik[6:8])
        month_digits = int(nik[8:10])
        year_short_digits = int(nik[10:12])

        day = day_digits - 40 if 41 <= day_digits <= 71 else day_digits

        current_year_short = datetime.now().year % 100
        # 对年份的判断稍微放宽一些，比如允许未来5年的年份（考虑到数据录入延迟等）
        year_full = 2000 + year_short_digits if year_short_digits <= current_year_short + 5 else 1900 + year_short_digits

        # 基本的日期有效性检查
        if not (1 <= month_digits <= 12): return None
        if not (1 <= day <= 31): return None  # 更细致的检查（如二月天数）可以在datetime构造时处理

        return datetime(year_full, month_digits, day).strftime("%Y-%m-%d")
    except ValueError:  # 转换int失败或日期无效
        return None


def normalize_date_from_text(date_text):
    if not isinstance(date_text, str): return None
    # 尝试匹配 dd-mm-yyyy
    match1 = re.match(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", date_text)
    if match1:
        day, month, year = match1.groups()
        try:  # 校验日期有效性
            return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
        except ValueError:
            pass  # 继续尝试其他格式

    # 尝试匹配 yyyy-mm-dd
    match2 = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", date_text)
    if match2:
        year, month, day = match2.groups()
        try:  # 校验日期有效性
            return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
        except ValueError:
            pass

    # 尝试匹配 yyyymmdd
    match3 = re.match(r"(\d{4})(\d{2})(\d{2})", date_text)
    if match3:
        year, month, day = match3.groups()
        try:
            return datetime(int(year), int(month), int(day)).strftime("%Y-%m-%d")
        except ValueError:
            pass

    return None  # 所有格式都不匹配或无效


# --- 主验证逻辑 ---
def verify_csv_data_with_minigpt4(csv_filepath, base_image_path="/home/risk/shuxiang/CNNPS/"):
    minigpt4_initialized = initialize_minigpt4()
    if not minigpt4_initialized:
        print("警告: MiniGPT-4 未能成功初始化。视觉性别判断将不可用。")

    results = []
    total_row_processing_time = 0
    total_llm_processing_time_overall = 0  # 累加所有LLM调用总耗时
    llm_processed_count = 0  # 成功调用LLM并获得有效性别（LAKI/PEREMPUAN/无法确定）的次数

    isps0_total_valid_benchmark_gender = 0
    isps0_llm_correct_count = 0
    isps1_llm_consistent_with_benchmark_count = 0
    isps1_total_valid_benchmark_gender = 0

    try:
        df = pd.read_csv(csv_filepath)
        print(f"成功读取CSV文件: {csv_filepath}，共 {len(df)} 条记录。")
    except FileNotFoundError:
        print(f"错误: CSV文件未找到于 {csv_filepath}");
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}");
        return

    with tqdm(total=df.shape[0], desc="处理进度", unit="条") as pbar:
        for index, row in df.iterrows():
            start_row_time = time.time()

            bid_no = row.get('bid_no', 'N/A')
            is_ps_raw = row.get('isPS', 'N/A')
            is_ps_status = "PS" if str(is_ps_raw) == '1' else "非PS" if str(is_ps_raw) == '0' else "isPS状态未知"

            id_card_no_csv_root = str(row.get('id_card_no', '')).strip()
            a001_txt_content = row.get('a001_txt_报文内容', '{}')

            relative_image_path_raw = str(row.get('a001_证件照片本地路径', '')).strip()
            full_image_path = ""
            if relative_image_path_raw:
                # 规范化路径，替换反斜杠，移除开头的 './'
                normalized_relative_path = relative_image_path_raw.replace("\\", "/")
                if normalized_relative_path.startswith('./'):
                    normalized_relative_path = normalized_relative_path[2:]
                full_image_path = os.path.join(base_image_path, normalized_relative_path)
                full_image_path = os.path.normpath(full_image_path)  # 确保路径格式正确 (e.g., /a/b/../c -> /a/c)

            record_result = {
                "bid_no": bid_no, "is_ps": is_ps_status,
                "原始图片路径_CSV": relative_image_path_raw if relative_image_path_raw else "路径未提供",
                "处理后图片路径": full_image_path if full_image_path else "路径未提供或为空",
                "身份证号_CSV根": id_card_no_csv_root if id_card_no_csv_root else "N/A",
                "身份证号_JSON": "N/A", "性别_JSON原始": "N/A", "性别_JSON规范化": "N/A",
                "性别_从NIK(CSV根)": "N/A", "性别_从NIK(JSON)": "N/A",
                "性别_视觉LLM": "未处理", "LLM性别判断使用的图像处理": "N/A", "LLM性别判断总耗时(秒)": 0.0,
                "匹配_JSON文本vsJSON_NIK": "未校验", "匹配_LLM视觉vsJSON_NIK": "未校验",
                "匹配_LLM视觉vsJSON文本": "未校验", "NIK一致性_根vsJSON": "未校验",
                "生日_JSON原始": "N/A", "生日_JSON规范化": "N/A",
                "生日_从NIK(CSV根)": "N/A", "生日_从NIK(JSON)": "N/A",
                "生日匹配_JSONvsJSON_NIK": "未校验",
                "姓名_CSV根": str(row.get('name', '')).strip() if pd.notna(row.get('name')) else "N/A",
                "姓名_JSON": "N/A", "姓名匹配_根vsJSON": "未校验",
                "单条总耗时(秒)": 0, "错误信息": []
            }

            try:
                json_data = json.loads(a001_txt_content)
            except json.JSONDecodeError:
                record_result["错误信息"].append("报文内容非有效JSON")
                json_data = {}  # 使用空字典以避免后续 .get() 出错

            id_number_from_json = str(json_data.get('id_number', '')).strip()
            gender_from_json_raw = json_data.get('gender', '')  # 可能为 None 或 ""
            birthday_from_json_raw = json_data.get('birthday', '')  # 可能为 None 或 ""
            name_from_json = str(json_data.get('name', '')).strip() if pd.notna(json_data.get('name')) else "N/A"

            record_result.update({
                "身份证号_JSON": id_number_from_json if id_number_from_json else "N/A",
                "性别_JSON原始": gender_from_json_raw if gender_from_json_raw else "N/A",
                "生日_JSON原始": birthday_from_json_raw if birthday_from_json_raw else "N/A",
                "姓名_JSON": name_from_json
            })

            if id_card_no_csv_root and id_number_from_json:
                record_result["NIK一致性_根vsJSON"] = "一致" if id_card_no_csv_root == id_number_from_json else "不一致"
            elif not id_card_no_csv_root and not id_number_from_json:
                record_result["NIK一致性_根vsJSON"] = "两者均为空"
            else:
                record_result["NIK一致性_根vsJSON"] = "其中一个为空"

            record_result["性别_JSON规范化"] = normalize_gender_from_text(gender_from_json_raw)
            if id_card_no_csv_root: record_result["性别_从NIK(CSV根)"] = get_gender_from_nik(id_card_no_csv_root)
            if id_number_from_json: record_result["性别_从NIK(JSON)"] = get_gender_from_nik(id_number_from_json)

            llm_gender_final = "未处理"
            llm_total_time_for_row = 0.0
            llm_attempt_desc = "N/A"

            if minigpt4_initialized and full_image_path and os.path.exists(full_image_path):
                try:
                    llm_gender_final, llm_total_time_for_row, llm_attempt_desc = get_visual_gender_with_minigpt4(
                        full_image_path)
                    record_result["性别_视觉LLM"] = llm_gender_final
                    record_result["LLM性别判断使用的图像处理"] = llm_attempt_desc
                    record_result["LLM性别判断总耗时(秒)"] = llm_total_time_for_row
                    total_llm_processing_time_overall += llm_total_time_for_row
                    # 仅当LLM实际处理了图像并返回了明确性别或“无法确定”时才计数（排除纯粹的图像加载/模型初始化错误）
                    if llm_gender_final in ["LAKI-LAKI", "PEREMPUAN", "无法确定_LLM"] or llm_gender_final.startswith(
                            "LLM图像") or llm_gender_final.startswith("LLM回答"):
                        llm_processed_count += 1
                except Exception as e_llm_call:  # 捕获 get_visual_gender_with_minigpt4 内部未捕获的意外错误
                    record_result["性别_视觉LLM"] = f"LLM调用意外错误: {str(e_llm_call)[:50]}"
                    record_result["错误信息"].append(f"LLM调用意外错误: {e_llm_call}")
            elif not minigpt4_initialized:
                record_result["性别_视觉LLM"] = "模型未初始化"
            elif not full_image_path:
                record_result["性别_视觉LLM"] = "图片路径CSV中为空"
            elif not os.path.exists(full_image_path):
                record_result["性别_视觉LLM"] = "图片未在处理后路径找到"

            g_json_norm = record_result["性别_JSON规范化"]
            g_nik_json = record_result["性别_从NIK(JSON)"]
            g_visual_llm = record_result["性别_视觉LLM"]

            benchmark_gender = "无法确定基准"
            # 优先使用 NIK 推断的性别作为基准
            if g_nik_json in ["LAKI-LAKI", "PEREMPUAN"]:
                benchmark_gender = g_nik_json
            # 其次使用 JSON 文本规范化后的性别
            elif g_json_norm in ["LAKI-LAKI", "PEREMPUAN"]:
                benchmark_gender = g_json_norm

            is_benchmark_gender_valid = benchmark_gender in ["LAKI-LAKI", "PEREMPUAN"]
            is_llm_gender_valid_for_comparison = g_visual_llm in ["LAKI-LAKI", "PEREMPUAN"]

            # 匹配_JSON文本vsJSON_NIK
            if g_json_norm not in ["无法识别_文本"] and g_nik_json not in ["NIK无效", "NIK日期码无效"]:
                record_result["匹配_JSON文本vsJSON_NIK"] = "一致" if g_json_norm == g_nik_json else "不一致"
            else:
                record_result["匹配_JSON文本vsJSON_NIK"] = "一方或双方无效/无法识别"

            # 匹配_LLM视觉vsJSON_NIK
            if is_llm_gender_valid_for_comparison and g_nik_json not in ["NIK无效", "NIK日期码无效"]:
                record_result["匹配_LLM视觉vsJSON_NIK"] = "一致" if g_visual_llm == g_nik_json else "不一致"
            else:
                record_result["匹配_LLM视觉vsJSON_NIK"] = "一方或双方无效/未处理"

            # 匹配_LLM视觉vsJSON文本
            if is_llm_gender_valid_for_comparison and g_json_norm not in ["无法识别_文本"]:
                record_result["匹配_LLM视觉vsJSON文本"] = "一致" if g_visual_llm == g_json_norm else "不一致"
            else:
                record_result["匹配_LLM视觉vsJSON文本"] = "一方或双方无效/未处理"

            # 统计LLM准确率 (isPS=0) 和一致性 (isPS=1)
            if is_ps_status == "非PS":  # isPS = 0
                if is_benchmark_gender_valid:
                    isps0_total_valid_benchmark_gender += 1
                    if is_llm_gender_valid_for_comparison and g_visual_llm == benchmark_gender:
                        isps0_llm_correct_count += 1
            elif is_ps_status == "PS":  # isPS = 1
                if is_benchmark_gender_valid:
                    isps1_total_valid_benchmark_gender += 1
                    if is_llm_gender_valid_for_comparison and g_visual_llm == benchmark_gender:
                        isps1_llm_consistent_with_benchmark_count += 1

            record_result["生日_JSON规范化"] = normalize_date_from_text(birthday_from_json_raw)
            if id_card_no_csv_root: record_result["生日_从NIK(CSV根)"] = get_birthdate_from_nik(id_card_no_csv_root)
            if id_number_from_json: record_result["生日_从NIK(JSON)"] = get_birthdate_from_nik(id_number_from_json)

            bd_json_norm = record_result["生日_JSON规范化"]
            bd_nik_json = record_result["生日_从NIK(JSON)"]
            if bd_json_norm and bd_nik_json:  # 两者都有效
                record_result["生日匹配_JSONvsJSON_NIK"] = "一致" if bd_json_norm == bd_nik_json else "不一致"
            elif not bd_json_norm and not bd_nik_json:  # 两者都无效
                record_result["生日匹配_JSONvsJSON_NIK"] = "两者均无效/无法规范化"
            else:  # 其中一个无效
                record_result["生日匹配_JSONvsJSON_NIK"] = "其中一个无效/无法规范化"

            name_csv = record_result["姓名_CSV根"]
            name_json_val = record_result["姓名_JSON"]
            if name_csv != "N/A" and name_json_val != "N/A":
                record_result["姓名匹配_根vsJSON"] = "一致" if name_csv.upper() == name_json_val.upper() else "不一致"
            elif name_csv == "N/A" and name_json_val == "N/A":
                record_result["姓名匹配_根vsJSON"] = "两者均为空/N/A"
            else:
                record_result["姓名匹配_根vsJSON"] = "其中一个为空/N/A"

            end_row_time = time.time()
            row_processing_time = end_row_time - start_row_time
            record_result["单条总耗时(秒)"] = round(row_processing_time, 3)
            total_row_processing_time += row_processing_time

            results.append(record_result)
            pbar.update(1)
            pbar.set_postfix_str(f"当前耗时: {row_processing_time:.3f}s, LLM耗时: {llm_total_time_for_row:.3f}s",
                                 refresh=True)

    processed_records = len(df) if not df.empty else 0

    output_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"验证结果_含MiniGPT4_{timestamp}.csv"
    try:
        output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n验证完成。结果已保存到: {output_filename}")
    except Exception as e:
        print(f"\n验证完成，但保存结果到CSV失败: {e}")
        try:
            # 尝试保存为txt作为备份
            backup_filename_txt = f"验证结果备份_{timestamp}.txt"
            with open(backup_filename_txt, "w", encoding="utf-8") as f_txt:
                f_txt.write(output_df.to_string())
            print(f"结果已作为文本保存在 {backup_filename_txt}")
        except Exception as e_txt:
            print(f"保存文本备份文件也失败: {e_txt}")

    print("\n--- 验证统计摘要 ---")
    if processed_records > 0:
        print(f"总处理记录数: {processed_records}")
        print(f"总处理耗时: {total_row_processing_time:.2f} 秒")
        print(f"平均每条记录处理耗时: {total_row_processing_time / processed_records:.3f} 秒")

        if llm_processed_count > 0:
            print(f"\n--- MiniGPT-4 性别判断统计 ---")
            print(f"参与LLM性别判断的记录数 (尝试调用LLM的次数): {llm_processed_count}")
            print(f"LLM性别判断总耗时 (所有参与LLM调用的记录): {total_llm_processing_time_overall:.2f} 秒")
            print(
                f"平均每次LLM性别判断耗时 (基于参与调用的记录): {total_llm_processing_time_overall / llm_processed_count:.3f} 秒")

            if isps0_total_valid_benchmark_gender > 0:
                llm_accuracy_isps0 = (isps0_llm_correct_count / isps0_total_valid_benchmark_gender) * 100
                print(
                    f"MiniGPT-4 性别识别准确率 (isPS=0, 基准有效样本数: {isps0_total_valid_benchmark_gender}): {llm_accuracy_isps0:.2f}% ({isps0_llm_correct_count}/{isps0_total_valid_benchmark_gender})")
            else:
                print("MiniGPT-4 性别识别准确率 (isPS=0): 无有效的基准样本进行计算。")

            if isps1_total_valid_benchmark_gender > 0:
                llm_consistency_isps1 = (
                                                    isps1_llm_consistent_with_benchmark_count / isps1_total_valid_benchmark_gender) * 100
                print(
                    f"MiniGPT-4 与基准性别一致性 (isPS=1, 基准有效样本数: {isps1_total_valid_benchmark_gender}): {llm_consistency_isps1:.2f}% ({isps1_llm_consistent_with_benchmark_count}/{isps1_total_valid_benchmark_gender})")
            else:
                print("MiniGPT-4 与基准性别一致性 (isPS=1): 无有效的基准样本进行计算。")
        else:
            print("\n--- MiniGPT-4 性别判断统计 ---")
            print("没有记录成功进行LLM性别判断或模型未初始化/图片路径无效。")

        print("\n--- 其他字段一致性统计 (基于输出结果) ---")
        mismatch_cols_info = {
            "匹配_JSON文本vsJSON_NIK": "性别 (JSON文本 vs JSON NIK)",
            "匹配_LLM视觉vsJSON_NIK": "性别 (LLM视觉 vs JSON NIK)",
            "匹配_LLM视觉vsJSON文本": "性别 (LLM视觉 vs JSON文本)",
            "NIK一致性_根vsJSON": "NIK (CSV根 vs JSON)",
            "生日匹配_JSONvsJSON_NIK": "生日 (JSON vs JSON NIK)",
            "姓名匹配_根vsJSON": "姓名 (CSV根 vs JSON)"
        }
        for col, desc in mismatch_cols_info.items():
            if col in output_df.columns:
                mismatches = output_df[output_df[col] == "不一致"]
                total_comparable = output_df[output_df[col].isin(["一致", "不一致"])]  # 计算可比较的总数
                mismatch_count = len(mismatches)
                total_comparable_count = len(total_comparable)
                percentage_str = f"({mismatch_count / total_comparable_count * 100:.2f}%)" if total_comparable_count > 0 else ""

                print(f"不一致 '{desc}': {mismatch_count} 条 / {total_comparable_count} 条可比较 {percentage_str}")
                if not mismatches.empty:
                    ps_mismatches = mismatches[mismatches['is_ps'] == "PS"]
                    non_ps_mismatches = mismatches[mismatches['is_ps'] == "非PS"]
                    # unknown_ps_mismatches = mismatches[mismatches['is_ps'] == "isPS状态未知"]
                    print(f"  -> 其中 PS: {len(ps_mismatches)}, 非PS: {len(non_ps_mismatches)}")
            else:
                print(f"警告: 列 '{col}' 在输出DataFrame中未找到，跳过其统计。")

        if "性别_视觉LLM" in output_df.columns:
            llm_uncertain = output_df[output_df["性别_视觉LLM"] == "无法确定_LLM"]
            print(f"MiniGPT-4 无法确定性别的记录数: {len(llm_uncertain)}")

            # 统计LLM处理过程中发生的具体错误类型
            llm_error_categories = {
                "图片加载/路径问题": ["图片未找到", "无法识别的图像格式", "图片加载错误", "图片路径CSV中为空",
                                      "图片未在处理后路径找到"],
                "LLM内部上传/转换问题": ["LLM图像上传失败", "LLM图像转换或编码失败", "LLM图像编码处理失败"],
                "LLM回答生成问题": ["LLM回答生成失败"],
                "模型初始化问题": ["模型或对话模板未初始化", "模型未初始化"],
                "LLM调用意外错误": ["LLM调用意外错误"]  # 捕获 get_visual_gender_with_minigpt4 中的意外错误
            }
            for error_desc, error_keywords in llm_error_categories.items():
                # 使用 str.contains 与 regex=True 来匹配包含任何一个关键字的行
                # na=False 确保 NaN 值不会导致错误，并且不被视为匹配
                # 使用 '|'.join() 来创建一个 OR 条件的正则表达式
                # 需要对关键字进行转义，以防它们包含正则表达式特殊字符 (虽然此例中不太可能)
                escaped_keywords = [re.escape(kw) for kw in error_keywords]
                error_count = output_df[
                    output_df["性别_视觉LLM"].str.contains("|".join(escaped_keywords), na=False, regex=True)].shape[0]
                if error_count > 0:
                    print(f"MiniGPT-4 因'{error_desc}'未能成功判断的记录数: {error_count}")
    else:
        print("没有数据被处理。")


if __name__ == "__main__":
    # 请根据您的实际路径修改这里的CSV文件路径和MiniGPT-4配置文件路径
    csv_file_to_process = "/home/risk/shuxiang/CNNPS/data/a001_processed_data.csv"  # 示例路径
    minigpt4_config_path_main = "eval_configs/minigpt4_eval.yaml"  # 示例路径
    base_image_directory = "/home/risk/shuxiang/CNNPS/"  # 示例图片基础路径

    if not os.path.exists(csv_file_to_process):
        print(f"错误: 输入的CSV文件 '{csv_file_to_process}' 未找到。请检查路径。")
        exit()

    if not os.path.exists(minigpt4_config_path_main):
        print(f"警告: MiniGPT-4 配置文件 '{minigpt4_config_path_main}' 未找到。")
        print("MiniGPT-4 的视觉性别判断功能可能无法使用或使用默认配置。")
        # 可以选择在这里退出，或者允许程序继续（initialize_minigpt4 会尝试使用默认路径）
        # exit()

    verify_csv_data_with_minigpt4(csv_file_to_process, base_image_path=base_image_directory)