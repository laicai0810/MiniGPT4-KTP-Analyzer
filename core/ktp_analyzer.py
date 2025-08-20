# -*- coding: utf-8 -*-
"""
ktp_analyzer.py
# ... (注释保持不变) ...
V11 (基于用户需求调整):
1.  parse_ktp_ocr 中对 Jenis Kelamin 字段进行严格规范化，仅输出 LAKI-LAKI 或 PEREMPUAN。
2.  对 Agama, Status Perkawinan, Kewarganegaraan 等字段引入类似的规范化逻辑。
3.  框架和方法名保持不变。
"""

import re
import numpy as np
from PIL import Image
import cv2


# ... (straighten_and_enhance_image 函数代码保持 V10 版本不变) ...
def straighten_and_enhance_image(image_pil):
    """
    自动对图像进行透视校正（拉直）。
    方法名 'straighten_and_enhance_image' 保留，但此版本仅执行透视校正。
    如果找不到清晰的四边形轮廓，则返回原始（可能已旋转的）图像。
    输出为RGB PIL Image。
    注意：此版本按照要求移除了 try-except 块。
    """
    # print("调用 straighten_and_enhance_image (仅执行透视校正)...")

    if image_pil.mode != 'RGB':
        image_pil_rgb = image_pil.convert('RGB')
    else:
        image_pil_rgb = image_pil.copy()

    open_cv_image_bgr = np.array(image_pil_rgb)
    open_cv_image_bgr = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_RGB2BGR)

    gray_for_contours = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_for_contours, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screen_cnt = None
    if open_cv_image_bgr.shape[0] > 30 and open_cv_image_bgr.shape[1] > 30:
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                if cv2.contourArea(approx) > (open_cv_image_bgr.shape[0] * open_cv_image_bgr.shape[1] * 0.1):
                    screen_cnt = approx
                    break
    else:
        pass

    output_pil_image = None
    if screen_cnt is not None:
        pts = screen_cnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        if maxWidth <= 0 or maxHeight <= 0:
            corrected_cv_image_rgb = cv2.cvtColor(open_cv_image_bgr, cv2.COLOR_BGR2RGB)
            output_pil_image = Image.fromarray(corrected_cv_image_rgb)
        else:
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                           dtype="float32")
            M = cv2.getPerspectiveTransform(rect, dst)
            warped_bgr = cv2.warpPerspective(open_cv_image_bgr, M, (maxWidth, maxHeight))
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            output_pil_image = Image.fromarray(warped_rgb)
    else:
        output_pil_image = image_pil_rgb

    return output_pil_image


# ----------------------------------------
# 2. OCR结果解析与字段对齐 (核心升级)
# ----------------------------------------
def parse_ktp_ocr(ocr_result):
    """
    重构的OCR解析函数，更鲁棒地提取KTP关键字段，能处理字段错位。
    增加字段值规范化逻辑。
    """
    lines = [line[1][0].strip() for line in ocr_result[0]] if ocr_result and ocr_result[0] else []

    parsed_data = {
        'provinsi': "N/A", 'kabupaten': "N/A", 'nik': "N/A", 'nama': "N/A",
        'tempat_tgl_lahir': "N/A", 'jenis_kelamin': "N/A", 'alamat': "N/A",
        'rt_rw': "N/A", 'kel_desa': "N/A", 'kecamatan': "N/A", 'agama': "N/A",
        'status_perkawinan': "N/A", 'pekerjaan': "N/A", 'kewarganegaraan': "N/A"
    }

    key_map = {
        'provinsi': [r"PROVINSI"], 'kabupaten': [r"KABUPATEN"],
        'nik': [r"NIK", r"N IK", r"N1K"],
        'nama': [r"Nama"], 'tempat_tgl_lahir': [r"Tempat/Tgl Lahir", r"Tempat/TglLahir", r"Tempat/Tgi Lahir"],
        'jenis_kelamin': [r"Jenis Kelamin", r"JenisKelamin", r"denis Kelamin"],  # 增加常见错误识别
        'alamat': [r"Alamat"],
        'rt_rw': [r"RT/RW"], 'kel_desa': [r"Kel/Desa", r"Kei/Desa"], 'kecamatan': [r"Kecamatan"],
        'agama': [r"Agama"], 'status_perkawinan': [r"Status Perkawinan", r"StatusPerkawinan", r"Status Perkawman"],
        'pekerjaan': [r"Pekerjaan"], 'kewarganegaraan': [r"Kewarganegaraan"]
    }

    line_consumed = [False] * len(lines)

    # --- 字段提取逻辑 (与V10基本一致) ---
    # 优先处理 NIK
    for i, line in enumerate(lines):
        if line_consumed[i]: continue
        for keyword in key_map['nik']:
            match = re.search(r"(?i)" + keyword + r"\s*[:\-\s]\s*([\d\sOQGSlBZEA]{16,})", line)
            if match:
                potential_nik = match.group(1).replace(" ", "")
                potential_nik = potential_nik.replace('O', '0').replace('Q', '0').replace('G', '6')
                potential_nik = potential_nik.replace('S', '5').replace('l', '1').replace('B', '8')
                potential_nik = potential_nik.replace('Z', '2').replace('E', '3').replace('A', '4')
                nik_digits = re.sub(r'\D', '', potential_nik)
                if len(nik_digits) == 16:
                    parsed_data['nik'] = nik_digits
                    line_consumed[i] = True
                    break
        if parsed_data['nik'] != "N/A": break  # NIK 已成功解析

    # 处理其他字段
    for field, keywords in key_map.items():
        if field == 'nik' and parsed_data['nik'] != "N/A": continue
        if parsed_data.get(field, "N/A") != "N/A" and field != 'jenis_kelamin':  # jenis_kelamin 需要后续特殊处理
            continue

        current_field_value = "N/A"  # 用于临时存储当前字段的提取值

        for i, line in enumerate(lines):
            if line_consumed[i]: continue
            for keyword in keywords:
                # 调整关键词匹配
                match = re.search(r"(?i)(^|\s)" + re.escape(keyword) + r"\s*[:\-\s]+\s*(.+)", line)
                value_on_same_line = ""
                keyword_found_on_line = False

                if match:
                    value_on_same_line = match.group(2).strip()
                    keyword_found_on_line = True
                elif re.search(r"(?i)^" + re.escape(keyword) + r"\s*$", line):  # 关键词占满一行
                    keyword_found_on_line = True

                if keyword_found_on_line:
                    is_another_keyword_in_value = False
                    if value_on_same_line:
                        for kw_list_check in key_map.values():
                            for kw_text_check in kw_list_check:
                                # 确保不是检查自身关键词
                                if kw_text_check.lower() != keyword.lower() and \
                                        re.search(r"(?i)\b" + re.escape(kw_text_check) + r"\b", value_on_same_line):
                                    is_another_keyword_in_value = True;
                                    break
                            if is_another_keyword_in_value: break

                    if value_on_same_line and not is_another_keyword_in_value:
                        current_field_value = value_on_same_line
                        line_consumed[i] = True;
                        break
                    elif i + 1 < len(lines) and not line_consumed[i + 1]:
                        next_line_is_keyword = False
                        for kw_list_check_next in key_map.values():
                            for kw_text_check_next in kw_list_check_next:
                                if re.search(r"(?i)(?:^|\s)" + re.escape(kw_text_check_next) + r"\s*[:\-\s]",
                                             lines[i + 1]):
                                    next_line_is_keyword = True;
                                    break
                            if next_line_is_keyword: break

                        if not next_line_is_keyword:
                            current_field_value = lines[i + 1].strip()
                            line_consumed[i] = True;
                            line_consumed[i + 1] = True;
                            break
            if current_field_value != "N/A":  # 如果当前字段已找到值
                parsed_data[field] = current_field_value  # 先存入初步提取的值
                break

    # --- NIK 后处理和补救措施 (与V10一致) ---
    full_text_no_space_upper = "".join(lines).replace(" ", "").upper()
    if parsed_data['nik'] == 'N/A' or len(re.sub(r'\D', '', parsed_data['nik'])) < 16:
        full_text_corrected_for_nik = full_text_no_space_upper.replace('O', '0').replace('Q', '0').replace('G', '6')
        full_text_corrected_for_nik = full_text_corrected_for_nik.replace('S', '5').replace('L', '1').replace('I',
                                                                                                              '1').replace(
            'B', '8')
        full_text_corrected_for_nik = full_text_corrected_for_nik.replace('Z', '2').replace('E', '3').replace('A', '4')
        nik_match = re.search(r'(\d{16})', re.sub(r'\D', '', full_text_corrected_for_nik))
        if nik_match:
            parsed_data['nik'] = nik_match.group(1)
        else:
            potential_nik_blocks = re.findall(r'([0-9OQGSLIBZEA]{16})', full_text_no_space_upper)
            for block in potential_nik_blocks:
                temp_nik = block.replace('O', '0').replace('Q', '0').replace('G', '6')
                temp_nik = temp_nik.replace('S', '5').replace('L', '1').replace('I', '1').replace('B', '8')
                temp_nik = temp_nik.replace('Z', '2').replace('E', '3').replace('A', '4')
                temp_nik_digits = re.sub(r'\D', '', temp_nik)
                if len(temp_nik_digits) == 16:
                    try:
                        day_nik = int(temp_nik_digits[6:8]);
                        month_nik = int(temp_nik_digits[8:10]);
                        year_digits = int(temp_nik_digits[10:12])
                        from datetime import date  # 动态获取当年年份后两位
                        current_year_last_two_digits = date.today().year % 100
                        year_nik = (2000 + year_digits) if year_digits <= current_year_last_two_digits else (
                                    1900 + year_digits)
                        if 1 <= month_nik <= 12:
                            day_check = day_nik
                            if day_check > 40: day_check -= 40
                            if 1 <= day_check <= 31: parsed_data['nik'] = temp_nik_digits; break
                    except ValueError:
                        continue

                    # 确保最终NIK是16位数字，否则置为N/A
    if parsed_data.get('nik', 'N/A') != 'N/A':
        parsed_data['nik'] = re.sub(r'\D', '', parsed_data['nik'])
        if len(parsed_data['nik']) != 16:
            parsed_data['nik'] = "N/A"

    # --- 字段值规范化 ---

    # 1. 性别 (Jenis Kelamin) 规范化
    jk_value = parsed_data.get('jenis_kelamin', "N/A")
    if jk_value != "N/A" and jk_value != "":  # 确保有值才处理
        # 先移除可能尾随的血型信息
        jk_cleaned_for_bloodtype = re.split(r"GOL(?:ONGAN|\.)?\s*DARAH", jk_value, flags=re.IGNORECASE)[0].strip()
        jk_cleaned_upper = jk_cleaned_for_bloodtype.upper().replace('-', '').replace(' ', '').replace('.', '')

        if "LAKI" in jk_cleaned_upper:  # 涵盖 LAKI, LAKILAKI, LAKI-LAKI
            parsed_data['jenis_kelamin'] = "LAKI-LAKI"
        elif "PEREMPUAN" in jk_cleaned_upper:  # 涵盖 PEREMPUAN
            parsed_data['jenis_kelamin'] = "PEREMPUAN"
        else:  # 如果清理和去除干扰后，仍然无法识别为标准性别
            parsed_data['jenis_kelamin'] = "N/A"
    else:  # 如果初始提取就是N/A或空
        parsed_data['jenis_kelamin'] = "N/A"

    # 如果性别字段仍然是N/A，尝试从全文中补救 (这种补救也需要规范化)
    if parsed_data['jenis_kelamin'] == 'N/A':
        if "LAKI-LAKI" in full_text_no_space_upper or \
                "LAKILAKI" in full_text_no_space_upper or \
                re.search(r"LAKI\s*(?:-| )\s*LAKI", full_text_no_space_upper):  # 匹配 LAKI LAKI 或 LAKI-LAKI
            parsed_data['jenis_kelamin'] = "LAKI-LAKI"
        elif "PEREMPUAN" in full_text_no_space_upper:
            parsed_data['jenis_kelamin'] = "PEREMPUAN"

    # 2. 宗教 (Agama) 规范化
    agama_value = parsed_data.get('agama', "N/A")
    if agama_value != "N/A" and agama_value != "":
        agama_text_upper = agama_value.upper()
        # 定义标准宗教名称及其常见变体/OCR错误（键为标准名，值为变体列表）
        standard_religions = {
            "ISLAM": ["ISLAM"],
            "KRISTEN": ["KRISTEN", "PROTESTAN", "KRISTEM"],
            "KATOLIK": ["KATOLIK", "KATOLK"],
            "HINDU": ["HINDU"],
            "BUDHA": ["BUDHA", "BUDDHA", "BUDHAISME"],
            "KHONGHUCU": ["KHONGHUCU", "KONGHUCU", "KHONG HU CU", "KONG HU CU"]
        }
        found_religion_std = "N/A"
        for standard_name, variants in standard_religions.items():
            for variant in variants:
                # 使用 \b 来确保是整个单词匹配，避免部分匹配 (例如 ISLAMIST)
                if re.search(r"\b" + re.escape(variant) + r"\b", agama_text_upper):
                    found_religion_std = standard_name
                    break
            if found_religion_std != "N/A":
                break
        parsed_data['agama'] = found_religion_std
    else:
        parsed_data['agama'] = "N/A"

    # 3. 婚姻状况 (Status Perkawinan) 规范化
    status_value = parsed_data.get('status_perkawinan', "N/A")
    if status_value != "N/A" and status_value != "":
        status_text_upper = status_value.upper().replace(" ", "")  # 去空格方便匹配
        standard_statuses = {
            "BELUM KAWIN": ["BELUMKAWIN", "BELUMKAHWIN"],
            "KAWIN": ["KAWIN", "KAHWIN", "MENIKAH"],
            "CERAI HIDUP": ["CERAIHIDUP"],
            "CERAI MATI": ["CERAIMATI"]
        }
        found_status_std = "N/A"
        # 对于婚姻状况，文本通常比较固定，可以直接比较清理后的
        for standard_name, variants in standard_statuses.items():
            for variant in variants:
                if variant in status_text_upper:  # 包含即可
                    found_status_std = standard_name
                    break
            if found_status_std != "N/A":
                break
        parsed_data['status_perkawinan'] = found_status_std
    else:
        parsed_data['status_perkawinan'] = "N/A"

    # 4. 国籍 (Kewarganegaraan) 规范化
    kwg_value = parsed_data.get('kewarganegaraan', "N/A")
    if kwg_value != "N/A" and kwg_value != "":
        kwg_text_upper = kwg_value.upper().replace(" ", "").replace(".", "")
        if "WNI" in kwg_text_upper or "INDONESIA" in kwg_text_upper:
            parsed_data['kewarganegaraan'] = "WNI"
        elif "WNA" in kwg_text_upper:  # 如果需要支持WNA
            parsed_data['kewarganegaraan'] = "WNA"
        else:
            parsed_data['kewarganegaraan'] = "N/A"
    else:
        parsed_data['kewarganegaraan'] = "N/A"

    # 对于其他字段，如Provinsi, Kabupaten, Nama, Alamat, Pekerjaan，
    # 主要是通用清理，例如去除首尾多余空格，或统一大小写（如果需要）。
    # 这里的基本提取逻辑已经做了 .strip()。
    # 如果需要更复杂的清理，可以针对性添加。
    for key in ['provinsi', 'kabupaten', 'nama', 'alamat', 'rt_rw', 'kel_desa', 'kecamatan', 'pekerjaan',
                'tempat_tgl_lahir']:
        if parsed_data[key] != "N/A":
            parsed_data[key] = parsed_data[key].strip()  # 确保去除首尾空格
            # 可以考虑去除连续多个空格
            parsed_data[key] = re.sub(r'\s+', ' ', parsed_data[key])

    return parsed_data


# ... (validate_indonesian_ktp 和 format_gender_bilingual 函数代码保持 V10 版本不变) ...
def validate_indonesian_ktp(data, wilayah_db):
    validation_results = []
    nik = data.get('nik', 'N/A')
    if nik == 'N/A' or not nik.isdigit() or len(nik) != 16:
        validation_results.append("❌ **NIK号码无效**: 未提取到或非16位纯数字号码。")
        return validation_results, "N/A"

    birth_date_digits_in_nik = int(nik[6:8])
    # 注意：这里的 gender_from_nik 推断逻辑是基于NIK编码规则的
    gender_from_nik = "LAKI-LAKI" if 1 <= birth_date_digits_in_nik <= 31 else "PEREMPUAN"

    # 由于 parse_ktp_ocr 已经规范化了 jenis_kelamin，这里的比较会更直接
    gender_field_normalized = data.get('jenis_kelamin', 'N/A')

    if gender_field_normalized != "N/A":
        # 直接比较规范化后的值
        if gender_from_nik == gender_field_normalized:
            validation_results.append(f"✅ **性别交叉验证一致**: NIK推断与字段均为'{gender_from_nik}'。")
        else:
            validation_results.append(
                f"❌ **性别交叉验证不一致**: NIK推断为'{gender_from_nik}', 但字段为'{gender_field_normalized}'。")
    else:
        validation_results.append(
            f"⚠️ **性别字段未提取或无法规范化**: 无法进行性别交叉验证。NIK推断为'{gender_from_nik}'。")

    province_code, regency_code = nik[:2], nik[:4]
    wilayah_info = "N/A"
    if regency_code in wilayah_db:
        wilayah_info = wilayah_db[regency_code]
        validation_results.append(
            f"✅ **行政区划代码有效 (Kab/Kota)**: NIK代码 '{regency_code}' 对应地区: {wilayah_info}。")
    elif province_code in wilayah_db:
        wilayah_info = wilayah_db[province_code]
        validation_results.append(
            f"✅ **行政区划代码有效 (Provinsi)**: NIK代码 '{province_code}' 对应地区: {wilayah_info}。")
    else:
        validation_results.append(
            f"⚠️ **行政区划代码未验证**: NIK代码 '{regency_code}' (或省份 '{province_code}') 未在本地数据库中找到。")

    provinsi_field, kabupaten_field = data.get('provinsi', 'N/A'), data.get('kabupaten', 'N/A')
    if wilayah_info != "N/A":  # 仅当NIK的行政区划有效时才进行交叉比对
        # 将字段值转为大写进行不区分大小写的比较
        provinsi_field_upper = provinsi_field.upper() if provinsi_field != "N/A" else "N/A"
        kabupaten_field_upper = kabupaten_field.upper() if kabupaten_field != "N/A" else "N/A"
        wilayah_info_upper = wilayah_info.upper()

        if provinsi_field_upper != "N/A" and kabupaten_field_upper != "N/A":
            if provinsi_field_upper in wilayah_info_upper and kabupaten_field_upper in wilayah_info_upper:
                validation_results.append(f"✅ **行政区划交叉验证一致 (省市均匹配)**。")
            else:
                validation_results.append(
                    f"❌ **行政区划交叉验证不一致**: NIK地区 '{wilayah_info}', 字段 '{provinsi_field} - {kabupaten_field}'。")
        elif provinsi_field_upper != "N/A":
            if provinsi_field_upper in wilayah_info_upper:
                validation_results.append(f"✅ **省级行政区划交叉验证一致**。")
            else:
                validation_results.append(
                    f"❌ **省级行政区划交叉验证不一致**: NIK地区为 '{wilayah_info}', 省份字段为 '{provinsi_field}'。")
        elif kabupaten_field_upper != "N/A":
            if kabupaten_field_upper in wilayah_info_upper:
                validation_results.append(f"✅ **市级行政区划交叉验证一致**。")
            else:
                validation_results.append(
                    f"❌ **市级行政区划交叉验证不一致**: NIK地区为 '{wilayah_info}', 市/县字段为 '{kabupaten_field}'。")

    day_nik_raw = int(nik[6:8])
    day_nik_adjusted = day_nik_raw - 40 if gender_from_nik == "PEREMPUAN" and day_nik_raw > 40 else day_nik_raw
    month_nik, year_digits_nik = int(nik[8:10]), int(nik[10:12])
    from datetime import date
    current_year_last_two_digits = date.today().year % 100
    year_nik_full = (2000 + year_digits_nik) if year_digits_nik <= current_year_last_two_digits else (
                1900 + year_digits_nik)

    try:
        from datetime import datetime
        datetime(year_nik_full, month_nik, day_nik_adjusted)
        nik_derived_date_str = f"{day_nik_adjusted:02d}-{month_nik:02d}-{year_nik_full}"
        validation_results.append(f"✅ **出生日期有效 (来自NIK)**: '{nik_derived_date_str}'。")

        tgl_lahir_field = data.get('tempat_tgl_lahir', 'N/A')
        if tgl_lahir_field != 'N/A':
            # 尝试从字段中提取 DD-MM-YYYY 格式的日期 (更灵活的月份和分隔符)
            date_match = re.search(r'(\d{1,2})[/\-\s.,]+(\d{1,2}|[A-Za-z]+)[/\-\s.,]+(\d{4})', tgl_lahir_field)
            if date_match:
                day_field_str, month_field_str, year_field_str = date_match.groups()
                day_field, year_field = int(day_field_str), int(year_field_str)

                month_map_indo_to_num = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MEI': 5, 'JUN': 6,
                    'JUL': 7, 'AGU': 8, 'SEP': 9, 'OKT': 10, 'NOV': 11, 'DES': 12,
                    'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4,  # MEI 已经在上面
                    'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8, 'SEPTEMBER': 9, 'OKTOBER': 10,
                    'NOVEMBER': 11, 'DESEMBER': 12
                }
                month_field = -1
                if month_field_str.isdigit():
                    month_field = int(month_field_str)
                else:  # 处理文字月份，取前三个字母并大写
                    month_field = month_map_indo_to_num.get(month_field_str[:3].upper(), -1)

                if month_field != -1 and 1 <= day_field <= 31 and 1 <= month_field <= 12:  # 基本的日期数字范围检查
                    try:  # 再次校验字段日期是否有效
                        datetime(year_field, month_field, day_field)
                        field_date_str = f"{day_field:02d}-{month_field:02d}-{year_field}"
                        if field_date_str == nik_derived_date_str:
                            validation_results.append(
                                f"✅ **出生日期交叉验证一致**: NIK日期与字段日期 ('{field_date_str}')匹配。")
                        else:
                            validation_results.append(
                                f"❌ **出生日期交叉验证不一致**: NIK推断日期为 '{nik_derived_date_str}', 但字段提取日期为 '{field_date_str}' (来自'{tgl_lahir_field}')。")
                    except ValueError:
                        validation_results.append(
                            f"⚠️ **出生日期字段解析后日期无效**: 字段为 '{tgl_lahir_field}'。NIK推断日期 '{nik_derived_date_str}'。")
                else:
                    validation_results.append(
                        f"⚠️ **出生日期字段格式无法完全解析或日期数字无效**: 字段为 '{tgl_lahir_field}'，月份 '{month_field_str}' 或日期 '{day_field_str}' 处理后不符合预期。NIK推断日期 '{nik_derived_date_str}'。")
            else:
                validation_results.append(
                    f"⚠️ **出生日期字段格式不标准**: 字段为 '{tgl_lahir_field}'。NIK推断日期 '{nik_derived_date_str}'。")
    except ValueError:
        validation_results.append(
            f"❌ **出生日期无效 (来自NIK)**: NIK中的日期部分 ('{day_nik_adjusted}-{month_nik}-{year_nik_full}') 不是一个有效的日期组合。")
    return validation_results, gender_from_nik


def format_gender_bilingual(gender_text):
    # 此函数现在接收的 gender_text 应该是已经规范化的 "LAKI-LAKI" 或 "PEREMPUAN" 或 "N/A"
    if gender_text == "LAKI-LAKI":
        return "LAKI-LAKI (男性)"
    elif gender_text == "PEREMPUAN":
        return "PEREMPUAN (女性)"
    return "N/A"  # 或 gender_text 如果可能传入其他值