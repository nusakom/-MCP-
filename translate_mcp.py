# -*- coding: utf-8 -*-
import json
import re
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模块级配置常量 ---
SUPPORTED_LANGS = {
    'en': '英语',
    'zh': '中文',
    'ja': '日语',
    'ko': '韩语',
    'fr': '法语',
    'de': '德语',
    'es': '西班牙语',
    'ru': '俄语',
    'it': '意大利语',
    'pt': '葡萄牙语'
}

SUPPORTED_STYLES = ['default', 'formal', 'casual', 'technical', 'literary']

DOMAIN_KEYWORDS = {
    "医学": ["医学", "病人", "症状", "诊断", "治疗", "药物", "手术", "医院", "医生", "护士"],
    "法律": ["法律", "合同", "诉讼", "法院", "律师", "条款", "权利", "义务", "法规", "判决"],
    "金融": ["金融", "投资", "股票", "基金", "银行", "利率", "货币", "资产", "负债", "风险"],
    "技术": ["技术", "软件", "硬件", "编程", "算法", "数据", "系统", "网络", "开发", "代码"],
    "教育": ["教育", "学习", "教学", "学生", "老师", "课程", "学校", "知识", "培训", "考试"]
}

COMMON_WORDS = ["the", "and", "for", "that", "this", "with", "你的", "我们", "他们"]
# --- 结束模块级配置常量 ---
"""
MCP（Multi-Channel Protocol）多通道翻译处理模块
本模块遵循MCP协议，支持结构化输入输出，便于智能助手（如Claude、ChatGPT等）直接集成。

API接口文档：
-------------------
1. 类：TranslateMCP
   - 功能：多通道翻译、领域分析、术语提取、分段翻译、全文审校、质量评估。
   - 支持多语言、风格与格式保持、术语一致性。

2. 主要方法：
   - register_channel(name, func): 注册翻译通道。
   - translate(text, target_lang, channel=None): 单句翻译。
   - analyze_and_plan(text): 结构化分析输入。
   - extract_terms(text): 术语提取。
   - segment_translate(segments, target_lang, channel=None): 分段翻译。
   - review_fulltext(translated_segments, style="default"): 合并审校。
   - quality_assess(src, tgt, terms=None): 质量评估。
   - translate_full_workflow(text, target_lang, channel=None, style="default"): 一站式MCP标准API，结构化输入输出。

3. 结构化输入输出示例：
   输入：{"text": "你好，世界！", "target_lang": "en", "channel": "fake", "style": "formal"}
   输出：{"analysis": {...}, "terms": [...], "translated_segments": [...], "reviewed": "...", "quality": {...}}
-------------------
"""

class TranslateMCP:
    """
    多通道翻译处理模块（MCP）
    支持输入原文、目标语言，返回结构化翻译结果。
    遵循MCP协议，便于AI助手集成。
    """
    def __init__(self):
        self.channels = {}
        # 使用模块级配置
        self.supported_langs = SUPPORTED_LANGS
        self.supported_styles = SUPPORTED_STYLES

    def register_channel(self, name, func):
        """
        注册新的翻译通道
        :param name: 通道名称
        :param func: 翻译函数，签名为 func(text, target_lang) -> str
        """
        self.channels[name] = func

    def translate(self, text, target_lang, channel=None):
        """
        翻译接口（单句/单段）
        :param text: 原文
        :param target_lang: 目标语言代码，如'en', 'zh', 'ja'等
        :param channel: 指定通道（可选）
        :return: 翻译结果字符串
        :raises: ValueError - 当参数无效或通道不存在时
        """
        if not text or not target_lang:
            raise ValueError('text和target_lang不能为空')
        
        # 验证目标语言是否支持
        if target_lang not in self.supported_langs:
            supported_keys = ', '.join(self.supported_langs.keys())
            logging.error(f'不支持的目标语言: {target_lang}. 支持的语言: {supported_keys}')
            raise ValueError(f'不支持的目标语言: {target_lang}。支持的语言: {supported_keys}')
            
        if channel:
            if channel not in self.channels:
                registered_channels = list(self.channels.keys())
                logging.error(f'未注册通道: {channel}. 已注册通道: {registered_channels}')
                raise ValueError(f'未注册通道: {channel}。已注册通道: {registered_channels}')
            try:
                return self.channels[channel](text, target_lang)
            except Exception as e:
                logging.error(f'通道 {channel} 调用错误: {e}', exc_info=True)
                return f"[翻译错误] {str(e)}"
        return f"[未实现翻译] {text} ({target_lang})"

    def analyze_and_plan(self, text):
        """
        分析规划阶段：自动识别文本领域、结构和翻译需求。
        :param text: 原文
        :return: dict，包括领域、结构、建议分段、语言检测等
        """
        if not text:
            return {"domain": "未知", "segments": [], "suggestion": "无", "language": "未知", "complexity": "简单"}
            
        domain = "通用"
        for d, keywords in DOMAIN_KEYWORDS.items():
            if any(word in text for word in keywords):
                domain = d
                break
                
        # 增加对中文、日文、韩文等标点的支持
        segments = re.split(r"[。.!?！？\n；;]", text)
        segments = [seg.strip() for seg in segments if seg.strip()]
        
        lang = self._detect_language(text)
        complexity = self._assess_complexity(text, segments)
            
        return {
            "domain": domain, 
            "segments": segments, 
            "suggestion": "按句分段", 
            "language": lang,
            "complexity": complexity,
            "total_length": len(text),
            "segment_count": len(segments)
        }

    def extract_terms(self, text, lang=None):
        """
        领域术语识别：提取关键术语并确保术语一致性。
        :param text: 原文
        :param lang: 文本语言，如果为None则自动检测
        :return: dict，包含术语列表及其分类
        """
        if not text or not isinstance(text, str):
            return {"terms": [], "categories": {}}
        import re
        
        # 自动检测语言
        if lang is None:
            # 简单语言检测：根据中文字符比例判断
            cn_char_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
            if cn_char_count / max(len(text), 1) > 0.5:
                lang = 'zh'
            else:
                lang = 'en'
                
        # 支持中英文术语提取（增强版）
        en_terms = re.findall(r"\b[A-Za-z][A-Za-z-]{2,}\b", text)
        zh_terms = re.findall(r"[\u4e00-\u9fa5]{2,}", text)
        num_terms = re.findall(r"\b\d+[.\-_]\d+[.\-\_\w]*\b", text)
        mixed_terms = re.findall(r"\b[A-Za-z\d]+[\-_][A-Za-z\d]+\b", text)
        
        # 根据语言调整术语提取优先级
        if lang == 'zh':
            # 中文文本优先提取中文术语
            all_terms = list(set(zh_terms + mixed_terms + en_terms + num_terms))
        else:
            # 英文文本优先提取英文术语
            all_terms = list(set(en_terms + mixed_terms + zh_terms + num_terms))
            
        categories = {
            "en": list(set(en_terms)),
            "zh": list(set(zh_terms)),
            "numeric": list(set(num_terms)),
            "mixed": list(set(mixed_terms))
        }
        
        # 添加术语频率信息
        term_freq = {term: text.count(term) for term in all_terms}
        filtered_terms = [term for term in all_terms if term.lower() not in COMMON_WORDS]
        
        # 按频率排序
        filtered_terms.sort(key=lambda x: term_freq[x], reverse=True)
        
        return {"terms": filtered_terms, "categories": categories, "frequencies": term_freq}

    def segment_translate(self, segments, target_lang, channel=None):
        """
        分段翻译：对分段文本逐段翻译。
        :param segments: 分段列表
        :param target_lang: 目标语言
        :param channel: 指定通道
        :return: 翻译后分段列表
        """
        return [self.translate(seg, target_lang, channel) for seg in segments if seg.strip()]

    def review_fulltext(self, translated_segments, style="default"):
        """
        全文审校：合并分段并根据风格调整表达。
        :param translated_segments: 翻译后分段列表
        :param style: 风格类型（如formal、casual、technical、literary、default等）
        :return: 审校后文本
        :raises: ValueError - 当风格类型不支持时
        """
        if not translated_segments or not isinstance(translated_segments, list):
            return ""
        if style not in self.supported_styles:
            supported_styles_list = ', '.join(self.supported_styles)
            logging.error(f'不支持的风格类型: {style}. 支持的风格: {supported_styles_list}')
            raise ValueError(f'不支持的风格类型: {style}。支持的风格: {supported_styles_list}')
        text = "。".join([str(seg) for seg in translated_segments if seg])
        if style == "formal":
            text = text.replace("你", "您").replace("觉得", "认为").replace("很好", "良好")
        elif style == "casual":
            text = text.replace("您", "你").replace("认为", "觉得").replace("非常感谢", "谢谢")
        elif style == "technical":
            text = text.replace("用", "使用").replace("做", "执行").replace("看", "查看")
        elif style == "literary":
            text = text.replace("很", "颇为").replace("非常", "极为")
        text = text.replace("。。", "。")
        text = text.replace("  ", " ")
        return text

    def quality_assess(self, src, tgt, terms=None):
        """
        质量评估：评估准确性、流畅性、术语和风格一致性。
        :param src: 原文
        :param tgt: 译文
        :param terms: 术语列表或术语字典
        :return: 评估报告dict，包含详细指标和建议
        """
        import re
        if not src or not tgt or not isinstance(src, str) or not isinstance(tgt, str):
            return {"error": "源文本或目标文本为空", "score": 0}
        score = 90
        missed_terms = []
        term_list = []
        if terms:
            if isinstance(terms, dict) and "terms" in terms:
                term_list = terms["terms"]
            elif isinstance(terms, list):
                term_list = terms
            missed_terms = [t for t in term_list if t not in tgt]
            score -= len(missed_terms) * 2
        src_len = len(re.findall(r'\w+|[\u4e00-\u9fff]', src))
        tgt_len = len(re.findall(r'\w+|[\u4e00-\u9fff]', tgt))
        length_ratio = tgt_len / src_len if src_len > 0 else 0
        if length_ratio < 0.5 or length_ratio > 2.0:
            score -= 10
        src_marks = re.findall(r'[,.!?;:，。！？；：]', src)
        tgt_marks = re.findall(r'[,.!?;:，。！？；：]', tgt)
        if abs(len(src_marks) - len(tgt_marks)) > 3:
            score -= 5
        accuracy = "高" if score > 85 else "中" if score > 70 else "低"
        fluency = "优秀" if score > 85 else "良好" if score > 75 else "一般" if score > 60 else "较差"
        terms_consistency = "完全一致" if not missed_terms else \
                           "基本一致" if len(missed_terms) <= 2 else \
                           "部分不一致" if len(missed_terms) <= 5 else "严重不一致"
        suggestions = []
        if missed_terms:
            suggestions.append(f"需要保持术语一致性，特别是: {', '.join(missed_terms[:3])}等")
        if length_ratio < 0.7:
            suggestions.append("译文可能过于简略，建议补充内容")
        elif length_ratio > 1.5:
            suggestions.append("译文可能过于冗长，建议精简")
        if score < 70:
            suggestions.append("整体质量需要提高，建议重新审校")
        return {
            "accuracy": accuracy, 
            "fluency": fluency, 
            "terms_consistency": terms_consistency, 
            "score": score,
            "missed_terms": missed_terms[:5] if missed_terms else [],
            "length_ratio": round(length_ratio, 2),
            "suggestions": suggestions
        }
        
    def _detect_language(self, text):
        """简单的语言检测函数"""
        if not text:
            return "unknown"
            
        # 计算中文字符比例
        cn_char_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
        cn_ratio = cn_char_count / max(len(text), 1)
        
        # 计算英文字符比例
        en_char_count = len(re.findall(r'[a-zA-Z]', text))
        en_ratio = en_char_count / max(len(text), 1)
        
        # 计算日文字符比例
        jp_char_count = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))
        jp_ratio = jp_char_count / max(len(text), 1)
        
        # 根据字符比例判断语言
        if cn_ratio > 0.3:
            return "zh"
        elif jp_ratio > 0.3:
            return "ja"
        elif en_ratio > 0.3:
            return "en"
        else:
            # 默认返回英文
            return "en"
            
    def _assess_complexity(self, text, segments):
        """评估文本复杂度"""
        if not text or not segments:
            return "简单"
            
        # 计算平均句长
        avg_seg_len = sum(len(seg) for seg in segments) / max(len(segments), 1)
        
        # 计算词汇多样性
        words = re.findall(r'\w+|[\u4e00-\u9fa5]+', text)
        unique_words = set(words)
        diversity = len(unique_words) / max(len(words), 1)
        
        # 根据句长和词汇多样性评估复杂度
        if avg_seg_len > 50 and diversity > 0.7:
            return "高度复杂"
        elif avg_seg_len > 30 or diversity > 0.6:
            return "复杂"
        elif avg_seg_len > 15 or diversity > 0.5:
            return "中等"
        else:
            return "简单"
            
    def _check_repeated_words(self, text):
        """检查文本中的重复词组"""
        if not text:
            return []
            
        # 分词处理
        if self._detect_language(text) == "zh":
            # 中文按字符分词
            words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        else:
            # 英文按单词分词
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
            
        # 检查连续重复
        repeated = []
        for i in range(len(words) - 1):
            if words[i] == words[i+1] and len(words[i]) > 1:
                repeated.append(words[i])
                
        return list(set(repeated))

    def translate_full_workflow(self, text, target_lang, channel=None, style="default", use_multi_channel=False):
        """
        MCP标准三阶段完整流程入口，结构化输入输出，适配AI助手API调用。
        遵循MCP协议的完整翻译流程：分析规划 -> 术语提取 -> 分段翻译 -> 全文审校 -> 质量评估
        
        :param text: 原文文本
        :param target_lang: 目标语言代码，如'en', 'zh', 'ja'等
        :param channel: 指定翻译通道，如果未指定则使用默认通道
        :param style: 风格类型，支持'default', 'formal', 'casual', 'technical', 'literary'
        :param use_multi_channel: 是否启用多通道融合翻译
        :return: dict，包含各阶段结构化结果，符合MCP协议规范
        :raises: ValueError - 当参数无效时
        """
        # 参数验证
        if not text:
            raise ValueError("原文不能为空")
        if not target_lang:
            raise ValueError("目标语言不能为空")
        if target_lang not in self.supported_langs:
            supported_keys = ', '.join(self.supported_langs.keys())
            logging.error(f'translate_full_workflow: 不支持的目标语言: {target_lang}. 支持的语言: {supported_keys}')
            raise ValueError(f'不支持的目标语言: {target_lang}。支持的语言: {supported_keys}')
        if style not in self.supported_styles:
            supported_styles_list = ', '.join(self.supported_styles)
            logging.error(f'translate_full_workflow: 不支持的风格类型: {style}. 支持的风格: {supported_styles_list}')
            raise ValueError(f'不支持的风格类型: {style}。支持的风格: {supported_styles_list}')
            
        start_time = time.time()
        process_log = []
        
        try:
            # 第一阶段：分析规划
            analysis = self.analyze_and_plan(text)
            process_log.append({"stage": "analyze", "time": time.time() - start_time})
            stage_time = time.time()
            
            # 第二阶段：术语提取
            terms = self.extract_terms(text)
            process_log.append({"stage": "extract_terms", "time": time.time() - stage_time})
            stage_time = time.time()
            
            # 获取分段
            segs = analysis["segments"]
            if not segs:
                return {
                    "analysis": analysis,
                    "terms": terms,
                    "translated_segments": [],
                    "reviewed": "",
                    "quality": {"error": "无可翻译内容", "score": 0},
                    "warning": "输入文本无法分段，请检查原文",
                    "process_time": round(time.time() - start_time, 3),
                    "status": "failed"
                }
                
            # 检测源文本语言
            source_lang = analysis.get("language", self._detect_language(text))
            analysis["language"] = source_lang
                
            # 第三阶段：分段翻译
            if use_multi_channel and len(self.channels) > 1:
                # 多通道融合翻译
                translated_results = self._multi_channel_translate(segs, target_lang)
                translated_segs = translated_results["segments"]
                channel_info = translated_results["channel_info"]
                analysis["multi_channel"] = channel_info
            else:
                # 单通道翻译
                translated_segs = self.segment_translate(segs, target_lang, channel)
                
            process_log.append({"stage": "translate", "time": time.time() - stage_time})
            stage_time = time.time()
            
            # 第四阶段：全文审校
            reviewed = self.review_fulltext(translated_segs, style)
            process_log.append({"stage": "review", "time": time.time() - stage_time})
            stage_time = time.time()
            
            # 第五阶段：质量评估
            qa = self.quality_assess(text, reviewed, terms)
            process_log.append({"stage": "assess", "time": time.time() - stage_time})
            
            # 返回符合MCP协议的结构化结果
            return {
                "analysis": analysis, 
                "terms": terms, 
                "translated_segments": translated_segs, 
                "reviewed": reviewed, 
                "quality": qa,
                "mcp_version": "1.0.0",
                "process_stages": ["analyze", "extract_terms", "translate", "review", "assess"],
                "process_time": {
                    "total": round(time.time() - start_time, 3),
                    "stages": process_log
                },
                "status": "success"
            }
        except Exception as e:
            logging.error(f'translate_full_workflow 发生错误: {e}', exc_info=True)
            import traceback
            error_info = traceback.format_exc()
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_details": error_info,
                "status": "failed",
                "process_time": round(time.time() - start_time, 3),
                "fallback": self._generate_fallback_translation(text, target_lang) if text and len(text) < 1000 else None
            }
            
    def _multi_channel_translate(self, segments, target_lang):
        """
        多通道融合翻译，综合多个翻译通道的结果
        :param segments: 分段列表
        :param target_lang: 目标语言
        :return: 融合后的翻译结果
        """
        if not segments:
            return {"segments": [], "channel_info": {}}
            
        # 获取可用的翻译通道
        available_channels = list(self.channels.keys())
        if not available_channels:
            return {"segments": segments, "channel_info": {"error": "无可用翻译通道"}}
            
        # 选择最多3个通道进行翻译
        selected_channels = available_channels[:3]
        channel_results = {}
        
        # 对每个通道进行翻译
        for channel in selected_channels:
            try:
                channel_results[channel] = self.segment_translate(segments, target_lang, channel)
            except Exception as e:
                logging.error(f"通道 {channel} 翻译失败: {e}")
                channel_results[channel] = None
                
        # 过滤掉失败的通道
        valid_results = {k: v for k, v in channel_results.items() if v is not None}
        if not valid_results:
            return {"segments": segments, "channel_info": {"error": "所有翻译通道均失败"}}
            
        # 融合翻译结果
        fused_segments = []
        channel_usage = {channel: 0 for channel in valid_results.keys()}
        
        for i, segment in enumerate(segments):
            # 收集所有通道对当前段落的翻译结果
            segment_translations = {}
            for channel, results in valid_results.items():
                if i < len(results):
                    segment_translations[channel] = results[i]
            
            # 选择最佳翻译结果
            if not segment_translations:
                fused_segments.append(segment)  # 如果没有有效翻译，使用原文
                continue
                
            # 简单策略：选择长度适中的翻译结果
            best_channel = self._select_best_translation(segment, segment_translations)
            fused_segments.append(segment_translations[best_channel])
            channel_usage[best_channel] += 1
            
        return {
            "segments": fused_segments,
            "channel_info": {
                "channels_used": list(valid_results.keys()),
                "channel_usage": channel_usage
            }
        }
        
    def _select_best_translation(self, source, translations):
        """
        从多个翻译结果中选择最佳的一个
        :param source: 源文本
        :param translations: 各通道的翻译结果字典
        :return: 最佳通道名称
        """
        if not translations:
            return None
            
        if len(translations) == 1:
            return list(translations.keys())[0]
            
        # 计算源文本长度
        source_len = len(source)
        
        # 计算每个翻译结果的长度比例
        ratios = {}
        for channel, translation in translations.items():
            ratio = len(translation) / max(source_len, 1)
            ratios[channel] = ratio
            
        # 选择长度比例最接近理想值的翻译
        # 理想比例：中译外1.5，外译中0.7，其他1.0
        source_lang = self._detect_language(source)
        target_lang = self._detect_language(list(translations.values())[0])
        
        if source_lang == 'zh' and target_lang != 'zh':
            ideal_ratio = 1.5
        elif source_lang != 'zh' and target_lang == 'zh':
            ideal_ratio = 0.7
        else:
            ideal_ratio = 1.0
            
        # 选择最接近理想比例的通道
        best_channel = min(ratios.keys(), key=lambda x: abs(ratios[x] - ideal_ratio))
        return best_channel
        
    def _generate_fallback_translation(self, text, target_lang):
        """
        生成应急翻译结果，在主翻译流程失败时使用
        :param text: 源文本
        :param target_lang: 目标语言
        :return: 应急翻译结果
        """
        try:
            # 尝试使用最简单的翻译方法
            if len(self.channels) > 0:
                # 使用第一个可用通道
                channel = list(self.channels.keys())[0]
                return self.translate(text, target_lang, channel)
            else:
                # 如果没有可用通道，返回简单替换结果
                return simple_translate(text, target_lang)
        except Exception as e:
            logging.error(f"应急翻译失败: {e}")
            return text  # 最坏情况下返回原文

    def mcp_api(self, request: dict) -> dict:
        """
        MCP标准API调用入口，结构化输入输出，适配AI助手。
        :param request: dict，包含text, target_lang, channel, style等
        :return: dict，结构化翻译结果
        :raises: ValueError - 当必要参数缺失或无效时
        """
        # 参数验证
        if not isinstance(request, dict):
            return {"error": "请求必须是字典格式", "status": "failed"}
            
        text = request.get("text")
        target_lang = request.get("target_lang")
        
        if not text:
            return {"error": "缺少必要参数: text", "status": "failed"}
        if not target_lang:
            return {"error": "缺少必要参数: target_lang", "status": "failed"}
            
        # 获取可选参数
        channel = request.get("channel")
        style = request.get("style", "default")
        use_multi_channel = request.get("use_multi_channel", False)
        
        # 验证目标语言
        if target_lang not in self.supported_langs:
            return {
                "error": f"不支持的目标语言: {target_lang}", 
                "supported_langs": list(self.supported_langs.keys()),
                "status": "failed"
            }
            
        # 验证风格
        if style not in self.supported_styles:
            return {
                "error": f"不支持的风格: {style}", 
                "supported_styles": self.supported_styles,
                "status": "failed"
            }
            
        # 验证通道
        if channel and channel not in self.channels:
            return {
                "error": f"未注册的翻译通道: {channel}", 
                "available_channels": list(self.channels.keys()),
                "status": "failed"
            }
        
        # 添加请求元数据
        metadata = {
            "request_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "MCP 1.0",
            "source_length": len(text),
            "params": {
                "target_lang": target_lang,
                "channel": channel,
                "style": style,
                "use_multi_channel": use_multi_channel
            }
        }
        
        # 检测源文本语言
        source_lang = self._detect_language(text)
        metadata["detected_language"] = source_lang
        
        # 检查文本长度限制
        if len(text) > 10000:  # 设置合理的长度限制
            return {
                "error": "文本长度超过限制",
                "max_length": 10000,
                "current_length": len(text),
                "status": "failed",
                "metadata": metadata
            }
        
        try:
            # 调用翻译全流程
            result = self.translate_full_workflow(
                text, 
                target_lang, 
                channel, 
                style,
                use_multi_channel
            )
            
            # 添加元数据
            result["metadata"] = metadata
            
            # 确保状态字段存在
            if "status" not in result:
                result["status"] = "success"
                
            return result
        except Exception as e:
            logging.error(f"MCP API调用失败: {e}", exc_info=True)
            import traceback
            error_info = traceback.format_exc()
            
            # 尝试生成应急翻译结果
            fallback = None
            if len(text) < 1000:  # 仅对短文本提供应急翻译
                try:
                    fallback = self._generate_fallback_translation(text, target_lang)
                except:
                    pass
                    
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "error_details": error_info,
                "status": "failed",
                "metadata": metadata,
                "fallback": fallback
            }

# 示例：注册几个简单的翻译通道
def fake_translate(text, target_lang):
    """伪翻译通道，仅用于测试"""
    # 支持多语言伪翻译
    return f"[Fake-{target_lang}] {text[::-1]}"
    
def echo_translate(text, target_lang):
    """回显通道，仅返回原文，用于测试"""
    return f"[Echo-{target_lang}] {text}"
    
def simple_translate(text, target_lang):
    """简单翻译通道，仅支持几个基本短语"""
    translations = {
        "en": {
            "你好": "Hello",
            "世界": "World",
            "谢谢": "Thank you",
            "再见": "Goodbye"
        },
        "ja": {
            "你好": "こんにちは",
            "世界": "世界",
            "谢谢": "ありがとう",
            "再见": "さようなら"
        },
        "fr": {
            "你好": "Bonjour",
            "世界": "Monde",
            "谢谢": "Merci",
            "再见": "Au revoir"
        }
    }
    
    if target_lang not in translations:
        return f"[不支持的语言-{target_lang}] {text}"
        
    # 简单替换已知短语
    result = text
    for zh, trans in translations[target_lang].items():
        result = result.replace(zh, trans)
        
    return result

def get_mcp_version():
    """返回MCP协议版本信息"""
    return {
        "name": "Multi-Channel Protocol Translation",
        "version": "1.0.0",
        "author": "MCP开发团队",
        "supported_languages": ["en", "zh", "ja", "ko", "fr", "de", "es", "ru", "it", "pt"],
        "supported_styles": ["default", "formal", "casual", "technical", "literary"],
        "api_spec": "https://example.com/mcp-api-spec"
    }

if __name__ == "__main__":
    mcp = TranslateMCP()
    # 注册多个翻译通道
    mcp.register_channel('fake', fake_translate)
    mcp.register_channel('echo', echo_translate)
    mcp.register_channel('simple', simple_translate)
    
    # 打印版本信息
    print("MCP协议翻译模块")
    print(f"版本: {get_mcp_version()['version']}")
    print(f"支持语言: {', '.join(mcp.supported_langs.keys())}")
    print(f"支持风格: {', '.join(mcp.supported_styles)}")
    print("-" * 50)
    
    # 标准API调用示例
    examples = [
        {"text": "你好，世界！", "target_lang": "en", "channel": "fake", "style": "formal"},
        {"text": "谢谢你的帮助。", "target_lang": "ja", "channel": "simple", "style": "casual"},
        {"text": "这是一个多通道翻译协议。", "target_lang": "fr", "channel": "echo", "style": "technical"}
    ]
    
    for i, request in enumerate(examples):
        print(f"\n示例 {i+1}:")
        print(f"输入: {request}")
        result = mcp.mcp_api(request)
        print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
    print("\n使用说明: 调用mcp_api方法，传入包含text和target_lang的字典，可选channel和style参数。")
    print("更多API文档请参考模块顶部注释。")