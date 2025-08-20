# **基于MiniGPT-4的多模态风险识别项目**

本项目是一个先进的多模态分析与风险校验系统，核心是利用强大的视觉语言模型 **MiniGPT-4** 对印尼身份证（KTP）图像进行深度分析，以实现自动化的信息提取和多维度风险校验。

## **✨ 功能特性**

* **智能图像预处理**:  
  * **透视校正**: 自动检测并拉直倾斜的身份证图像。  
  * **方向校正**: 利用人脸检测技术，自动旋转倒置或侧置的图像，确保文字识别的准确性。  
  * **图像增强**: 对图像进行锐化等处理，提升OCR识别率。  
* **高精度信息提取**:  
  * **OCR识别**: 使用PaddleOCR引擎，专门针对印尼语进行优化，从图像中提取所有文本信息。  
  * **字段解析**: 通过健壮的正则表达式和逻辑匹配，将OCR文本精准地解析并填充到“姓名”、“NIK”、“性别”等标准化字段中。  
* **自动化规则校验**:  
  * **交叉验证**: 自动比对从NIK号码中推断出的“性别”、“出生日期”、“行政区划”等信息与OCR识别出的字段内容是否一致。  
  * **数据库校验**: 将NIK中的行政区划代码与本地数据库进行比对，验证其有效性。  
* **多模态性别验证**:  
  * 利用 **MiniGPT-4** 的视觉理解能力，直接从证件照片中推断人物的表观性别。  
  * 将视觉推断结果与NIK推断结果进行交叉比对，为性别字段提供双重验证，极大地提升了识别的可靠性。  
* **数据预处理与微调**:  
  * 提供数据预处理脚本 (scripts/create\_finetune\_dataset.py)，可将原始数据自动转换为适用于MiniGPT-4微调的对话格式。

## **🚀 环境与安装**

### **1\. 核心依赖: MiniGPT-4**

本项目基于 **MiniGPT-4** 的官方实现。请首先按照其官方指南完成环境的搭建和预训练模型的下载。

* **MiniGPT-4 官方GitHub仓库**: [https://github.com/Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)

请务必确保您已成功下载 **Vicuna** 或 **Llama2** 的LLM权重，并正确配置了 eval\_configs/minigpt4\_eval.yaml 文件。

### **2\. 本项目依赖**

在MiniGPT-4环境搭建完成后，您还需要安装本项目特定的依赖库。

pip install \-r requirements.txt

## **📁 项目结构**

MiniGPT4-KTP-Analyzer/  
├── main\_app.py                 \# Gradio交互界面主程序  
├── core/  
│   └── ktp\_analyzer.py         \# 核心分析模块  
├── scripts/  
│   ├── batch\_verifier.py       \# 批量验证脚本  
│   └── create\_finetune\_dataset.py \# 微调数据生成脚本  
├── data/  
│   └── provinces.csv  
│   └── regencies.csv  
├── eval\_configs/  
│   └── minigpt4\_eval.yaml  
├── README.md  
├── requirements.txt  
└── .gitignore

## **📖 使用指南**

### **1\. 交互式分析 (Gradio WebUI)**

直接运行 main\_app.py 文件即可启动一个本地的Web界面。

\# 确保您的MiniGPT-4模型路径和配置正确  
python main\_app.py \--cfg-path eval\_configs/minigpt4\_eval.yaml \--gpu-id 0

* **专业证件照分析 (KTP)**: 在此标签页下，上传单张KTP图片，点击“开始智能分析”即可获得完整的结构化信息和校验报告。  
* **通用看图对话 (自由模式)**: 在此标签页下，您可以上传任意图片，与MiniGPT-4进行自由对话。

### **2\. 批量数据验证**

如果您需要对大量的KTP数据进行批量处理和验证，请使用 scripts/batch\_verifier.py 脚本。

1. **准备数据**: 准备一个包含图片路径和相关信息的CSV文件。  
2. **修改路径**: 打开脚本，在文件末尾的 if \_\_name\_\_ \== "\_\_main\_\_": 部分，修改文件和目录路径。  
3. **运行脚本**:  
   python scripts/batch\_verifier.py

   脚本运行结束后，会自动生成一个包含详细验证结果的CSV报告文件。

## **👨‍💻 作者**

* **杨书祥 (laicai0810)**  
* **GitHub**: [https://github.com/laicai0810](https://github.com/laicai0810)