"""
L2 筛选结果 PDF 报告生成
用法: python report_l2.py
"""
import json, os, sys
from datetime import date
from pathlib import Path

# UTF-8 输出
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), 'w', encoding='utf-8', buffering=1)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── 字体注册（Windows） ───────────────────────────────────────────
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, HRFlowable,
                                 PageBreak, KeepTogether)
from reportlab.lib.colors import HexColor

try:
    pdfmetrics.registerFont(TTFont('SimHei', 'C:/Windows/Fonts/simhei.ttf'))
    pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
    pdfmetrics.registerFont(TTFont('Consolas', 'C:/Windows/Fonts/consola.ttf'))
    FONT = 'SimHei'
    FONT_SIMSUN = 'SimSun'
    FONT_MONO = 'Consolas'
except Exception:
    FONT = 'Helvetica'
    FONT_SIMSUN = 'Helvetica'
    FONT_MONO = 'Courier'

# ── 颜色 ───────────────────────────────────────────────────────────
NAVY   = HexColor("#1C2B4B")
GOLD   = HexColor("#C9A84C")
LIGHT  = HexColor("#F5F7FA")
WHITE  = colors.white
GRAY   = HexColor("#6B7280")
GREEN  = HexColor("#059669")
ORANGE = HexColor("#D97706")
RED    = HexColor("#DC2626")
LIGHT_BLUE = HexColor("#DBEAFE")

# ── 样式 ───────────────────────────────────────────────────────────
def style(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

TITLE_STYLE   = style('title', fontName=FONT, fontSize=22, textColor=WHITE,
                        alignment=TA_CENTER, spaceAfter=6, leading=28)
SUBTITLE_STYLE = style('subtitle', fontName=FONT, fontSize=11, textColor=HexColor("#CBD5E1"),
                        alignment=TA_CENTER, spaceAfter=4, leading=14)
H1_STYLE      = style('h1', fontName=FONT, fontSize=14, textColor=NAVY,
                        spaceAfter=6, spaceBefore=12, leading=18)
BODY_STYLE    = style('body', fontName=FONT, fontSize=9, textColor=HexColor("#374151"),
                        spaceAfter=4, leading=12)
LABEL_STYLE   = style('label', fontName=FONT, fontSize=8, textColor=GRAY, leading=10)
FOOTER_STYLE  = style('footer', fontName=FONT_SIMSUN, fontSize=7, textColor=GRAY,
                        alignment=TA_CENTER)
NOTE_STYLE    = style('note', fontName=FONT, fontSize=7.5, textColor=GRAY,
                        spaceAfter=4, leading=10)

# ── 表格样式工厂 ────────────────────────────────────────────────────
def table_style(header_bg=NAVY, row_alt=LIGHT):
    ts = TableStyle([
        ('FONTNAME',      (0,0), (-1,0),  FONT),
        ('FONTSIZE',     (0,0), (-1,0),   8),
        ('FONTNAME',     (0,1), (-1,-1), FONT),
        ('FONTSIZE',     (0,1), (-1,-1), 7.5),
        ('BACKGROUND',   (0,0), (-1,0),  header_bg),
        ('TEXTCOLOR',    (0,0), (-1,0),  WHITE),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, row_alt]),
        ('GRID',         (0,0), (-1,-1), 0.4, HexColor("#E5E7EB")),
        ('TOPPADDING',   (0,0), (-1,-1), 4),
        ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ('LEFTPADDING',  (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        # 表头底部加粗分隔线
        ('LINEBELOW',    (0,0), (-1,0), 1.5, GOLD),
    ])
    return ts

# ── 优先级标签颜色 ─────────────────────────────────────────────────
PRIORITY_COLORS = {
    'both':  HexColor("#059669"),   # 两者都有 → 绿
    '1p':    HexColor("#1D4ED8"),   # 1p强力  → 蓝
    'vol':   HexColor("#7C3AED"),   # 量价齐升 → 紫
    '1':     HexColor("#374151"),   # 1类买点  → 深灰
}
PRIORITY_LABEL = {
    'both': '⓵ 两者皆有',
    '1p':   '② 1P强力型',
    'vol':  '③ 量价齐升',
    '1':    '④ 1类买点',
}

# ── 信号颜色 ────────────────────────────────────────────────────────
def sig_color(s):
    s = s or ''
    if '买(1);' in s and '量价齐升' in s: return PRIORITY_COLORS['both']
    if '买(1p)' in s: return PRIORITY_COLORS['1p']
    if '量价齐升' in s: return PRIORITY_COLORS['vol']
    if '买(1)' in s: return PRIORITY_COLORS['1']
    return GRAY

# ── 主程序 ─────────────────────────────────────────────────────────
def main():
    # 加载数据
    l2 = json.load(open(OUT_DIR / "l2_candidates.json", encoding="utf-8"))
    sl  = json.load(open(OUT_DIR / "stock_list.json",  encoding="utf-8"))['data']
    items = l2['data']
    today = str(date.today())

    # 排序（已在 report_l2.py 前一步排好）
    def sort_key(it):
        has_both = bool(it.get('chan_buy') and it.get('wave_bull'))
        has_1p   = '1p' in (it.get('chan_buy_detail') or '')
        has_vol  = '量价齐升' in (it.get('wave_bull_detail') or '')
        has_1    = bool(it.get('chan_buy') and '1p' not in (it.get('chan_buy_detail') or ''))
        if has_both: return 0
        if has_1p:   return 1
        if has_vol:  return 2
        if has_1:    return 3
        return 4
    items.sort(key=sort_key)

    # 统计
    n_both = sum(1 for i in items if i.get('chan_buy') and i.get('wave_bull'))
    n_1p   = sum(1 for i in items if '1p' in (i.get('chan_buy_detail') or ''))
    n_vol  = sum(1 for i in items if '量价齐升' in (i.get('wave_bull_detail') or '') and not (i.get('chan_buy')))
    n_1    = len(items) - n_both - n_1p - n_vol

    # ── PDF 文件 ────────────────────────────────────────────────────
    pdf_path = OUT_DIR / f"L2_选股报告_{today}.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        leftMargin=1.2*cm, rightMargin=1.2*cm,
        topMargin=1.5*cm,  bottomMargin=1.2*cm,
    )

    story = []

    # ══════════════════════════════════════════════════════════════
    # 封面标题栏
    # ══════════════════════════════════════════════════════════════
    title_data = [[
        Paragraph(f"<b>📈 L2 形态选股报告</b>", style(TITLE_STYLE.name, **{
            'fontName': FONT, 'fontSize': 20, 'textColor': WHITE,
            'alignment': TA_CENTER, 'leading': 26})),
    ]]
    title_tbl = Table(title_data, colWidths=[28*cm])
    title_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), NAVY),
        ('TOPPADDING',   (0,0), (-1,-1), 14),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('LEFTPADDING',  (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))

    sub_data = [[
        Paragraph(f"{today} · L1+L2 两层量化筛选 · 共 <b>{len(items)}</b> 只候选股",
                  style(SUBTITLE_STYLE.name, fontName=FONT, fontSize=10,
                        textColor=HexColor("#CBD5E1"), alignment=TA_CENTER, leading=14)),
    ]]
    sub_tbl = Table(sub_data, colWidths=[28*cm])
    sub_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), NAVY),
        ('TOPPADDING',  (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 12),
    ]))

    story.append(title_tbl)
    story.append(sub_tbl)
    story.append(Spacer(1, 8))

    # ══════════════════════════════════════════════════════════════
    # 统计卡片行
    # ══════════════════════════════════════════════════════════════
    stat_cards = [
        ("🏆 两者皆有",    f"{n_both}", "缠论买点 + 量价齐升",  PRIORITY_COLORS['both']),
        ("💎 1P强力型",   f"{n_1p}",   "缠论1类强力买点",      PRIORITY_COLORS['1p']),
        ("📊 量价齐升",    f"{n_vol}",   "仅波浪信号",          PRIORITY_COLORS['vol']),
        ("🔖 1类买点",    f"{n_1}",    "缠论1类普通买点",     PRIORITY_COLORS['1']),
        ("📋 合计候选",    f"{len(items)}", "L2 通过总数",        NAVY),
    ]
    card_cells = []
    card_width = 28 * cm / len(stat_cards)
    for label, val, desc, c in stat_cards:
        cell_data = [[
            Paragraph(f"<b>{label}</b>", style('sc_l', fontName=FONT, fontSize=8,
                        textColor=c, alignment=TA_CENTER, leading=10)),
            Paragraph(f"<font size=16 color='#{c.hexval()[2:]}'>{val}</font>",
                      style('sc_v', fontName=FONT, fontSize=16, textColor=c,
                            alignment=TA_CENTER, leading=20)),
            Paragraph(f"<font size=7 color='#6B7280'>{desc}</font>",
                      style('sc_d', fontName=FONT, fontSize=7, textColor=GRAY,
                            alignment=TA_CENTER, leading=9)),
        ]]
        ct = Table(cell_data, colWidths=[card_width - 4*mm])
        ct.setStyle(TableStyle([
            ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',  (0,0), (-1,-1), 4),
            ('BOTTOMPADDING',(0,0), (-1,-1), 4),
        ]))
        card_cells.append(ct)

    card_row = Table([card_cells], colWidths=[card_width] * len(stat_cards))
    card_row.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), LIGHT),
        ('BOX',          (0,0), (-1,-1), 0.5, HexColor("#E5E7EB")),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LEFTPADDING',  (0,0), (-1,-1), 4),
        ('RIGHTPADDING',(0,0), (-1,-1), 4),
    ]))
    story.append(card_row)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════
    # 主表格
    # ══════════════════════════════════════════════════════════════
    headers = ["优先级", "代码", "名称", "缠论买点", "波浪信号", "L1条件",
               "现价(元)", "PE", "总市值(亿)", "备注"]

    # 分配列宽
    col_widths = [2.0*cm, 1.8*cm, 2.8*cm, 4.0*cm, 5.5*cm,
                  1.4*cm, 1.8*cm, 1.6*cm, 2.0*cm, 3.5*cm]

    tbl_data = [headers]
    for i, it in enumerate(items):
        code = it['code']
        name = sl.get(code, {}).get('name', code) if isinstance(sl, dict) else code
        cond = ''.join([t for t in ['A','B','C'] if it.get(f'cond_{t.lower()}')])
        chan = it.get('chan_buy_detail', '') or ('有买点' if it.get('chan_buy') else '—')
        wave = it.get('wave_bull_detail', '') or '—'
        pe   = f"{it['pe']:.1f}" if it.get('pe') else 'N/A'
        close = f"{it.get('close', 0):.2f}"
        mktcap = f"{it.get('mktcap', 0):.0f}" if it.get('mktcap') else 'N/A'

        # 优先级
        has_both = bool(it.get('chan_buy') and it.get('wave_bull'))
        has_1p   = '1p' in chan
        has_vol  = '量价齐升' in wave
        if has_both:
            prio_label = "① 两者皆有"
            prio_color = PRIORITY_COLORS['both']
        elif has_1p:
            prio_label = "② 1P强力型"
            prio_color = PRIORITY_COLORS['1p']
        elif has_vol:
            prio_label = "③ 量价齐升"
            prio_color = PRIORITY_COLORS['vol']
        else:
            prio_label = "④ 1类买点"
            prio_color = PRIORITY_COLORS['1']

        # PE 颜色
        pe_val = it.get('pe')
        if pe_val and pe_val < 0:
            pe_color = RED
        elif pe_val and pe_val > 100:
            pe_color = ORANGE
        else:
            pe_color = HexColor("#374151")

        row = [
            Paragraph(f"<font color='#{prio_color.hexval()[2:]}'>{prio_label}</font>",
                      style(f'p{i}', fontName=FONT, fontSize=7, alignment=TA_CENTER, leading=9)),
            Paragraph(f"<b>{code}</b>",
                      style(f'c{i}', fontName=FONT_MONO, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph(f"<b>{name}</b>",
                      style(f'n{i}', fontName=FONT, fontSize=7.5, alignment=TA_LEFT)),
            Paragraph(f"<font color='#{sig_color(chan).hexval()[2:]}'>{chan}</font>",
                      style(f'ch{i}', fontName=FONT, fontSize=7, alignment=TA_CENTER, leading=9)),
            Paragraph(f"<font color='#{sig_color(wave).hexval()[2:]}'>{wave}</font>",
                      style(f'w{i}', fontName=FONT, fontSize=7, alignment=TA_CENTER, leading=9)),
            Paragraph(f"<b>{cond}</b>",
                      style(f't{i}', fontName=FONT, fontSize=8, alignment=TA_CENTER)),
            Paragraph(f"<b>{close}</b>",
                      style(f'cl{i}', fontName=FONT, fontSize=8, alignment=TA_CENTER)),
            Paragraph(f"<font color='#{pe_color.hexval()[2:]}'>{pe}</font>",
                      style(f'pe{i}', fontName=FONT, fontSize=8, alignment=TA_CENTER)),
            Paragraph(mktcap,
                      style(f'mc{i}', fontName=FONT, fontSize=7.5, alignment=TA_CENTER)),
            Paragraph("—",
                      style(f'note{i}', fontName=FONT, fontSize=7, alignment=TA_CENTER,
                            textColor=GRAY)),
        ]
        tbl_data.append(row)

    main_tbl = Table(tbl_data, colWidths=col_widths, repeatRows=1)
    ts = table_style()
    # 优先级列背景
    for i, it in enumerate(items):
        has_both = bool(it.get('chan_buy') and it.get('wave_bull'))
        has_1p   = '1p' in (it.get('chan_buy_detail') or '')
        has_vol  = '量价齐升' in (it.get('wave_bull_detail') or '') and not it.get('chan_buy')
        if has_both:
            bg = HexColor("#D1FAE5")
        elif has_1p:
            bg = HexColor("#DBEAFE")
        elif has_vol:
            bg = HexColor("#EDE9FE")
        else:
            bg = WHITE if i % 2 == 0 else LIGHT
        ts.add('BACKGROUND', (0, i+1), (0, i+1), bg)
    main_tbl.setStyle(ts)
    story.append(main_tbl)
    story.append(Spacer(1, 8))

    # ── 信号说明 ────────────────────────────────────────────────────
    legend_data = [[
        Paragraph(f"<font color='#059669'>●</font> ① 两者皆有 = 缠论1类买点 + 量价齐升  "
                  f"<font color='#1D4ED8'>●</font> ② 1P强力型 = 缠论1类强力买点  "
                  f"<font color='#7C3AED'>●</font> ③ 量价齐升 = 仅波浪信号  "
                  f"<font color='#374151'>●</font> ④ 1类买点 = 缠论1类普通买点  "
                  f"<font color='#DC2626'>●</font> PE负值 = 亏损  "
                  f"<font color='#D97706'>●</font> PE>100 = 极高估值",
                  style('leg', fontName=FONT, fontSize=7.5, textColor=GRAY, leading=11)),
    ]]
    leg_tbl = Table(legend_data, colWidths=[28*cm])
    leg_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), LIGHT),
        ('BOX',          (0,0), (-1,-1), 0.4, HexColor("#E5E7EB")),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
    ]))
    story.append(leg_tbl)

    # ── 免责声明 ─────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    disclaimer = ("本报告仅供参考，不构成投资建议。股市有风险，入市须谨慎。"
                  "所有数据来源于通达信本地行情和公开市场信息，量化模型存在局限性，"
                  "过往业绩不代表未来表现。")
    story.append(Paragraph(disclaimer, NOTE_STYLE))

    # ── 生成 ────────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF 已生成: {pdf_path}")
    return str(pdf_path)


if __name__ == "__main__":
    main()
