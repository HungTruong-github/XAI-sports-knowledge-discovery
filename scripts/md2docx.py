"""Chuyển docs/*.md sang .docx để nộp thầy.

Hỗ trợ: heading, đoạn văn, bảng GFM, danh sách gạch đầu dòng / đánh số,
khối mã, đường kẻ ngang, và inline **đậm** / *nghiêng* / `mã`.
"""

import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

BOLD = re.compile(r"(\*\*.+?\*\*)")
CODE_ITALIC = re.compile(r"(`.+?`|\*[^*]+?\*)")


def add_runs(paragraph, text, bold=False):
    """Tách text thành các run theo cú pháp inline của markdown.

    Xử lý lồng nhau: `mã` và *nghiêng* nằm bên trong **đậm** vẫn được nhận.
    """
    for part in BOLD.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_runs(paragraph, part[2:-2], bold=True)
            continue
        for sub in CODE_ITALIC.split(part):
            if not sub:
                continue
            if sub.startswith("`") and sub.endswith("`"):
                run = paragraph.add_run(sub[1:-1])
                run.font.name = "Consolas"
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
            elif sub.startswith("*") and sub.endswith("*"):
                run = paragraph.add_run(sub[1:-1])
                run.italic = True
            else:
                run = paragraph.add_run(sub)
            run.bold = bold or None


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_separator(line):
    return bool(re.fullmatch(r"\|[\s:|-]+\|", line.strip()))


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=len(rows), cols=len(header))
    table.style = "Table Grid"
    for j, cell in enumerate(header):
        para = table.rows[0].cells[j].paragraphs[0]
        add_runs(para, cell)
        for run in para.runs:
            run.bold = True
    for i, row in enumerate(body, start=1):
        for j in range(len(header)):
            para = table.rows[i].cells[j].paragraphs[0]
            add_runs(para, row[j] if j < len(row) else "")
    doc.add_paragraph()


def convert(md_path, docx_path):
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Khối mã
        if stripped.startswith("```"):
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            para = doc.add_paragraph()
            run = para.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            i += 1
            continue

        # Bảng
        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            rows = [split_row(stripped)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # Đường kẻ ngang
        if stripped == "---":
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # Trích dẫn (blockquote) — gộp các dòng liên tiếp, thụt lề và in nghiêng
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.4)
            para.paragraph_format.space_before = Pt(6)
            para.paragraph_format.space_after = Pt(6)
            add_runs(para, " ".join(buf))
            for run in para.runs:
                run.italic = True
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            head = doc.add_heading("", level=min(level, 4))
            add_runs(head, m.group(2))
            for run in head.runs:
                run.font.name = "Times New Roman"
                run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            if level == 1:
                head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # Danh sách gạch đầu dòng (kể cả checkbox)
        m = re.match(r"^(\s*)[-*]\s+(?:\[[ x]\]\s+)?(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            para = doc.add_paragraph(style="List Bullet" if indent == 0 else "List Bullet 2")
            add_runs(para, m.group(2))
            i += 1
            continue

        # Danh sách đánh số
        m = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            para = doc.add_paragraph(style="List Number" if indent == 0 else "List Number 2")
            add_runs(para, m.group(2))
            i += 1
            continue

        # Đoạn văn: gộp các dòng liên tiếp
        buf = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("```")
                or nxt == "---"
                or re.match(r"^\s*([-*]|\d+\.)\s", lines[i])
            ):
                break
            buf.append(nxt)
            i += 1
        add_runs(doc.add_paragraph(), " ".join(buf))

    doc.save(docx_path)
    print(f"Da xuat: {docx_path}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
