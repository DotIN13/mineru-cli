# Register models
from mineru.model import vlm_hf_model as _
from mineru.model import vlm_sglang_model as _

# Load MinerU
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import load_images_from_pdf
from mineru.backend.vlm.base_predictor import BasePredictor
from mineru.backend.vlm.vlm_analyze import ModelSingleton

from mineru_cli.token_to_middle_json import result_to_middle_json


def vlm_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: BasePredictor | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    f_dump_table_images: bool = True,
    f_dump_interline_equation_images: bool = True,
    **kwargs,
):
    if predictor is None:
        predictor = ModelSingleton().get_model(
            backend, model_path, server_url, **kwargs
        )

    # Load images from PDF
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
    images_base64_list = [image_dict["img_base64"] for image_dict in images_list]

    # Perform inference
    results = predictor.batch_predict(images=images_base64_list)

    # Convert results to middle JSON format
    middle_json = result_to_middle_json(
        results,
        images_list,
        pdf_doc,
        image_writer,
        f_dump_table_images=f_dump_table_images,
        f_dump_interline_equation_images=f_dump_interline_equation_images,
    )
    return middle_json, results
