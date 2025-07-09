import time
import cv2
import numpy as np
from loguru import logger

from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.utils.config_reader import get_llm_aided_config
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType
from mineru.utils.hash_utils import str_md5
from mineru.backend.vlm.vlm_magic_model import MagicModel
from mineru.utils.llm_aided import llm_aided_title
from mineru.utils.pdf_image_tools import get_crop_img
from mineru.version import __version__


def token_to_page_info(
    token,
    image_dict,
    page,
    image_writer,
    page_index,
    f_dump_table_images,
    f_dump_interline_equation_images,
) -> dict:
    """将token转换为页面信息"""
    # 解析token，提取坐标和类型
    # 假设token格式为：<|box_start|>x0 y0 x1 y1<|box_end|><|ref_start|>type<|ref_end|><|md_start|>content<|md_end|>
    # 这里需要根据实际的token格式进行解析
    # 提取所有完整块，每个块从<|box_start|>开始到<|md_end|>或<|im_end|>结束

    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = str_md5(image_dict["img_base64"])
    width, height = map(int, page.get_size())

    magic_model = MagicModel(token, width, height)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()

    # 如果有标题优化需求，则对title_blocks截图det
    llm_aided_config = get_llm_aided_config()
    if llm_aided_config is not None:
        title_aided_config = llm_aided_config.get("title_aided", None)
        if title_aided_config is not None:
            if title_aided_config.get("enable", False):
                atom_model_manager = AtomModelSingleton()
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name="ocr",
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang="ch_lite",
                )
                for title_block in title_blocks:
                    title_pil_img = get_crop_img(
                        title_block["bbox"], page_pil_img, scale
                    )
                    title_np_img = np.array(title_pil_img)
                    # 给title_pil_img添加上下左右各50像素白边padding
                    title_np_img = cv2.copyMakeBorder(
                        title_np_img,
                        50,
                        50,
                        50,
                        50,
                        cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    )
                    title_img = cv2.cvtColor(title_np_img, cv2.COLOR_RGB2BGR)
                    ocr_det_res = ocr_model.ocr(title_img, rec=False)[0]
                    if len(ocr_det_res) > 0:
                        # 计算所有res的平均高度
                        avg_height = np.mean(
                            [box[2][1] - box[0][1] for box in ocr_det_res]
                        )
                        title_block["line_avg_height"] = round(avg_height / scale)

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    image_types_to_save = [ContentType.IMAGE]
    if f_dump_table_images:
        image_types_to_save.append(ContentType.TABLE)
    if f_dump_interline_equation_images:
        image_types_to_save.append(ContentType.INTERLINE_EQUATION)

    all_spans = magic_model.get_all_spans()
    for span in all_spans:
        if span["type"] in image_types_to_save:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    page_blocks = []
    page_blocks.extend(
        [
            *image_blocks,
            *table_blocks,
            *title_blocks,
            *text_blocks,
            *interline_equation_blocks,
        ]
    )
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {
        "para_blocks": page_blocks,
        "discarded_blocks": [],
        "page_size": [width, height],
        "page_idx": page_index,
    }
    return page_info


def result_to_middle_json(
    token_list,
    images_list,
    pdf_doc,
    image_writer,
    f_dump_table_images=True,
    f_dump_interline_equation_images=True,
):
    middle_json = {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}
    for index, token in enumerate(token_list):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = token_to_page_info(
            token,
            image_dict,
            page,
            image_writer,
            index,
            f_dump_table_images,
            f_dump_interline_equation_images,
        )
        middle_json["pdf_info"].append(page_info)

    """llm优化"""
    llm_aided_config = get_llm_aided_config()

    if llm_aided_config is not None:
        """标题优化"""
        title_aided_config = llm_aided_config.get("title_aided", None)
        if title_aided_config is not None:
            if title_aided_config.get("enable", False):
                llm_aided_title_start_time = time.time()
                llm_aided_title(middle_json["pdf_info"], title_aided_config)
                logger.info(
                    f"llm aided title time: {round(time.time() - llm_aided_title_start_time, 2)}"
                )

    # 关闭pdf文档
    pdf_doc.close()
    return middle_json
