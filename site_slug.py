"""GitHub 风格的标题锚点 slugify（保留中文，与 GitHub 渲染保持一致）。"""

import re


def github_slugify(value, separator="-"):
    value = str(value).strip().lower()
    # 保留字母/数字/中文/空格/连字符/下划线，去掉其余标点
    value = re.sub(r"[^\w\s-]", "", value, flags=re.UNICODE)
    value = re.sub(r"[-\s]+", "-", value)
    return value.strip("-")
