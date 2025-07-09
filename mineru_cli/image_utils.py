from loguru import logger

from mineru.utils.pdf_image_tools import get_crop_img
from mineru.utils.pdf_reader import image_to_bytes
from mineru.utils.hash_utils import str_sha256
from mineru.data.data_reader_writer import FileBasedDataWriter


def cut_image(
    bbox: tuple,
    page_num: int,
    page_pil_img,
    return_path,
    image_writer: FileBasedDataWriter,
    scale: int = 2,
):
    """Crop the region defined by *bbox* from *page_pil_img* on *page_num* and save it as a JPEG.

    The saved file name is composed of the page number and an 8‑character prefix of the SHA‑256 hash of the legacy
    *img_path* for determinism, in the form:

        {page_num}_{hash8}.jpg

    The image bytes are persisted via *image_writer* (which can point to S3 or the local filesystem) and the relative
    path that was written is returned so callers can reconstruct a full URL if necessary.
    """

    # Build a deterministic legacy path (still used as the hash seed so existing callers get the same value)
    page_and_bbox = (
        f"{page_num}_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"
    )
    image_key = (
        f"{return_path}_{page_and_bbox}" if return_path is not None else page_and_bbox
    )

    # Generate a short hash (first 8 characters of SHA‑256)
    short_hash = str_sha256(image_key)[:8]
    final_filename = f"page_{page_num}_{short_hash}.jpg"

    # Crop the specified area and encode to JPEG bytes
    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)
    img_bytes = image_to_bytes(crop_img, image_format="JPEG")

    # Persist the image via the provided writer and return the relative file name
    image_writer.write(final_filename, img_bytes)
    return final_filename


def check_img_bbox(bbox) -> bool:
    if any([bbox[0] >= bbox[2], bbox[1] >= bbox[3]]):
        logger.warning(f"image_bboxes: 错误的box, {bbox}")
        return False

    return True


def cut_image_and_table(
    span, page_pil_img, page_img_md5, page_id, image_writer, scale=2
):
    if not check_img_bbox(span["bbox"]) or not image_writer:
        span["image_path"] = ""
    else:
        span["image_path"] = cut_image(
            span["bbox"],
            page_id,
            page_pil_img,
            return_path=f"{span['type']}/{page_img_md5}",
            image_writer=image_writer,
            scale=scale,
        )

    return span
