# draw_bbox.py

import json
import os
from pathlib import Path

import fitz  # PyMuPDF 1.25+

from mineru.utils.enum_class import BlockType, ContentType


def _normalize_color(rgb):
    """
    Convert an RGB list of 0-255 ints to 0-1 floats for PyMuPDF.
    """
    return [c / 255.0 for c in rgb]


def _transform_rect(bbox, page_height):
    """
    Transform a bbox from [x0, y0, x1, y1] with origin bottom-left
    to a fitz.Rect with origin top-left.
    """
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0, y0, x1, y1)


def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    """
    Draw layout-level bounding boxes on each PDF page using PyMuPDF.
    Filled boxes (e.g., tables, images) are drawn with 30% opacity.
    Numbering of blocks is drawn in red.
    """
    # Prepare lists of bboxes per page
    dropped_bbox_list = []
    tables_body_list = []
    tables_caption_list = []
    tables_footnote_list = []
    imgs_body_list = []
    imgs_caption_list = []
    imgs_footnote_list = []
    titles_list = []
    texts_list = []
    intereq_list = []
    lists_list = []
    index_list = []

    for page in pdf_info:
        dropped = []
        t_body = []
        t_caption = []
        t_foot = []
        i_body = []
        i_caption = []
        i_foot = []
        titles = []
        texts = []
        intereq = []
        lists_ = []
        idxs = []

        for blk in page.get('discarded_blocks', []):
            dropped.append(blk['bbox'])

        for blk in page.get('para_blocks', []):
            if blk['type'] == BlockType.TABLE:
                for sub in blk['blocks']:
                    if sub['type'] == BlockType.TABLE_BODY:
                        t_body.append(sub['bbox'])
                    elif sub['type'] == BlockType.TABLE_CAPTION:
                        t_caption.append(sub['bbox'])
                    elif sub['type'] == BlockType.TABLE_FOOTNOTE:
                        t_foot.append(sub['bbox'])
            elif blk['type'] == BlockType.IMAGE:
                for sub in blk['blocks']:
                    if sub['type'] == BlockType.IMAGE_BODY:
                        i_body.append(sub['bbox'])
                    elif sub['type'] == BlockType.IMAGE_CAPTION:
                        i_caption.append(sub['bbox'])
                    elif sub['type'] == BlockType.IMAGE_FOOTNOTE:
                        i_foot.append(sub['bbox'])
            elif blk['type'] == BlockType.TITLE:
                titles.append(blk['bbox'])
            elif blk['type'] == BlockType.TEXT:
                texts.append(blk['bbox'])
            elif blk['type'] == BlockType.INTERLINE_EQUATION:
                intereq.append(blk['bbox'])
            elif blk['type'] == BlockType.LIST:
                lists_.append(blk['bbox'])
            elif blk['type'] == BlockType.INDEX:
                idxs.append(blk['bbox'])

        dropped_bbox_list.append(dropped)
        tables_body_list.append(t_body)
        tables_caption_list.append(t_caption)
        tables_footnote_list.append(t_foot)
        imgs_body_list.append(i_body)
        imgs_caption_list.append(i_caption)
        imgs_footnote_list.append(i_foot)
        titles_list.append(titles)
        texts_list.append(texts)
        intereq_list.append(intereq)
        lists_list.append(lists_)
        index_list.append(idxs)

    # Flatten block order for numbering
    layout_bbox_list = []
    order_map = {BlockType.TABLE_CAPTION: 1, BlockType.TABLE_BODY: 2, BlockType.TABLE_FOOTNOTE: 3}
    for page in pdf_info:
        flat = []
        for blk in page.get('para_blocks', []):
            if blk['type'] in [BlockType.TEXT, BlockType.TITLE, BlockType.INTERLINE_EQUATION, BlockType.LIST, BlockType.INDEX]:
                flat.append(blk['bbox'])
            elif blk['type'] == BlockType.IMAGE:
                for sub in blk['blocks']:
                    flat.append(sub['bbox'])
            elif blk['type'] == BlockType.TABLE:
                subs = sorted(blk['blocks'], key=lambda s: order_map.get(s['type'], 0))
                for sub in subs:
                    flat.append(sub['bbox'])
        layout_bbox_list.append(flat)

    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')

    # Configs: (bbox_list, color, is_fill)
    configs = [
        (dropped_bbox_list, _normalize_color([158,158,158]), True),
        (tables_body_list,   _normalize_color([204,204,0]),   True),
        (tables_caption_list,_normalize_color([255,255,102]), True),
        (tables_footnote_list,_normalize_color([229,255,204]),True),
        (imgs_body_list,     _normalize_color([153,255,51]),  True),
        (imgs_caption_list,  _normalize_color([102,178,255]), True),
        (imgs_footnote_list, _normalize_color([255,178,102]), True),
        (titles_list,        _normalize_color([102,102,255]), True),
        (texts_list,         _normalize_color([153,0,76]),    True),
        (intereq_list,       _normalize_color([0,255,0]),     True),
        (lists_list,         _normalize_color([40,169,92]),   True),
        (index_list,         _normalize_color([40,169,92]),   True),
    ]

    for i, page in enumerate(doc):
        h = page.rect.height
        # Draw rectangles
        for lst, col, is_fill in configs:
            for bbox in lst[i]:
                rect = _transform_rect(bbox, h)
                if is_fill:
                    # filled with 30% opacity
                    page.draw_rect(rect, fill=col, fill_opacity=0.3)
                else:
                    # stroke only
                    page.draw_rect(rect, color=col, width=1)

        # Draw numbering with red
        num_col = _normalize_color([255,0,0])
        for j, bbox in enumerate(layout_bbox_list[i]):
            x0, y0, x1, y1 = bbox
            pos = fitz.Point(x1 + 2, y0 + 4)
            page.insert_text(pos, str(j+1), fontsize=10, color=num_col)

    # Ensure output dir
    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_path_full = os.path.join(out_path, filename)
    doc.save(out_path_full)
    doc.close()


def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    """
    Draw span-level bounding boxes (outline only) on each PDF page using PyMuPDF.
    """
    text_list = []
    inline_eq_list = []
    interline_eq_list = []
    image_list = []
    table_list = []
    dropped_list = []
    next_text = []
    next_inline = []

    def collect(span, target, next_target=None):
        if span.get('cross_page') and next_target is not None:
            next_target.append(span['bbox'])
        else:
            target.append(span['bbox'])

    for page in pdf_info:
        pt, pi, pI, pimg, ptbl, pd = [], [], [], [], [], []
        if next_text:
            pt.extend(next_text); next_text.clear()
        if next_inline:
            pi.extend(next_inline); next_inline.clear()

        # dropped spans
        for blk in page.get('discarded_blocks', []):
            if blk['type'] == BlockType.DISCARDED:
                for ln in blk.get('lines', []):
                    for sp in ln.get('spans', []):
                        pd.append(sp['bbox'])

        # other spans
        for blk in page.get('preproc_blocks', []):
            if blk['type'] in [BlockType.TEXT, BlockType.TITLE, BlockType.INTERLINE_EQUATION, BlockType.LIST, BlockType.INDEX]:
                for ln in blk.get('lines', []):
                    for sp in ln.get('spans', []):
                        if sp['type'] == ContentType.TEXT:
                            collect(sp, pt, next_text)
                        elif sp['type'] == ContentType.INLINE_EQUATION:
                            collect(sp, pi, next_inline)
                        elif sp['type'] == ContentType.INTERLINE_EQUATION:
                            pI.append(sp['bbox'])
            elif blk['type'] in [BlockType.IMAGE, BlockType.TABLE]:
                for sub in blk.get('blocks', []):
                    for ln in sub.get('lines', []):
                        for sp in ln.get('spans', []):
                            if sp['type'] == ContentType.IMAGE:
                                pimg.append(sp['bbox'])
                            elif sp['type'] == ContentType.TABLE:
                                ptbl.append(sp['bbox'])

        text_list.append(pt)
        inline_eq_list.append(pi)
        interline_eq_list.append(pI)
        image_list.append(pimg)
        table_list.append(ptbl)
        dropped_list.append(pd)

    # Open PDF
    doc = fitz.open(stream=pdf_bytes, filetype='pdf')

    # Colors and lists
    settings = [
        (_normalize_color([255,0,0]), text_list),
        (_normalize_color([0,255,0]), inline_eq_list),
        (_normalize_color([0,0,255]), interline_eq_list),
        (_normalize_color([255,204,0]), image_list),
        (_normalize_color([204,0,255]), table_list),
        (_normalize_color([158,158,158]), dropped_list),
    ]

    for i, page in enumerate(doc):
        h = page.rect.height
        for col, lst in settings:
            for bbox in lst[i]:
                rect = _transform_rect(bbox, h)
                page.draw_rect(rect, color=col, width=1)

    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(out_path, filename)
    doc.save(out_file)
    doc.close()


if __name__ == '__main__':
    pdf_path = 'data/test-files/mineru/28_questions/vlm/28_questions_origin.pdf'
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    json_path = 'data/test-files/mineru/28_questions/vlm/28_questions_middle.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    draw_layout_bbox(data['pdf_info'], pdf_bytes, 'examples', 'output_layout.pdf')
    draw_span_bbox(data['pdf_info'], pdf_bytes, 'examples', 'output_span.pdf')
