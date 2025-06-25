#!/usr/bin/env python3
"""
Command-line utility to parse documents using VLM backends (vlm-sglang-engine or vlm-sglang-client).
Supports individual files, directories, and glob patterns, with proper error handling.
"""
import os
os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

import json
import argparse
from pathlib import Path
import glob
from loguru import logger
from multiprocessing import Pool

# Register models
from mineru.model import vlm_hf_model as _
from mineru.model import vlm_sglang_model as _

# Load MinerU
from mineru.cli.common import prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

from mineru_cli.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru_cli.pdf_utils import extract_pdf_bytes_by_pymupdf


def do_parse(
    output_dir: Path,
    pdf_paths: list[Path],
    backend: str,
    server_url: str | None = None,
    f_draw_layout_bbox: bool = True,
    f_draw_span_bbox: bool = False,
    f_dump_md: bool = True,
    f_dump_middle_json: bool = True,
    f_dump_model_output: bool = True,
    f_dump_orig_pdf: bool = True,
    f_dump_content_list: bool = True,
    f_make_md_mode: MakeMode = MakeMode.MM_MD
) -> None:
    """
    Parse documents using VLM backends. Processes each file with isolation and logging.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        pdf_name = pdf_path.stem
        logger.info(f"Starting parse for '{pdf_name}'")
        try:
            # Read and preprocess
            pdf_bytes = read_fn(pdf_path)
            processed_bytes = extract_pdf_bytes_by_pymupdf(pdf_bytes, 0, None)

            # Prepare output dirs
            image_dir, md_dir = prepare_env(output_dir, pdf_name, 'vlm')
            image_writer = FileBasedDataWriter(image_dir)
            md_writer = FileBasedDataWriter(md_dir)

            # Analyze via VLM
            middle_json, infer_output = vlm_doc_analyze(
                processed_bytes,
                image_writer=image_writer,
                backend=backend,
                server_url=server_url
            )
            pdf_info = middle_json.get('pdf_info', {})

            # Draw bounding boxes
            if f_draw_layout_bbox:
                try:
                    draw_layout_bbox(pdf_info, processed_bytes, md_dir, f"{pdf_name}_layout.pdf")
                except Exception as e:
                    logger.error(f"Layout bbox failed for {pdf_name}: {e}")
            if f_draw_span_bbox:
                try:
                    draw_span_bbox(pdf_info, processed_bytes, md_dir, f"{pdf_name}_span.pdf")
                except Exception as e:
                    logger.error(f"Span bbox failed for {pdf_name}: {e}")

            # Write outputs with individual error handling
            if f_dump_orig_pdf:
                try:
                    md_writer.write(f"{pdf_name}_origin.pdf", processed_bytes)
                except Exception as e:
                    logger.error(f"Failed writing original PDF for {pdf_name}: {e}")
            if f_dump_md:
                try:
                    img_folder = os.path.basename(image_dir)
                    md_content = vlm_union_make(pdf_info, f_make_md_mode, img_folder)
                    md_writer.write_string(f"{pdf_name}.md", md_content)
                except Exception as e:
                    logger.error(f"Markdown dump failed for {pdf_name}: {e}")
            if f_dump_content_list:
                try:
                    img_folder = os.path.basename(image_dir)
                    content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, img_folder)
                    md_writer.write_string(
                        f"{pdf_name}_content_list.json",
                        json.dumps(content_list, ensure_ascii=False, indent=4)
                    )
                except Exception as e:
                    logger.error(f"Content list dump failed for {pdf_name}: {e}")
            if f_dump_middle_json:
                try:
                    md_writer.write_string(
                        f"{pdf_name}_middle.json",
                        json.dumps(middle_json, ensure_ascii=False, indent=4)
                    )
                except Exception as e:
                    logger.error(f"Middle JSON dump failed for {pdf_name}: {e}")
            if f_dump_model_output:
                try:
                    raw_output = "\n".join(infer_output)
                    md_writer.write_string(f"{pdf_name}_model_output.txt", raw_output)
                except Exception as e:
                    logger.error(f"Model output dump failed for {pdf_name}: {e}")

            logger.info(f"Completed parse for '{pdf_name}'")

        except Exception as main_e:
            logger.exception(f"Unexpected error processing {pdf_name}: {main_e}")


def expand_inputs(inputs: list[str]) -> list[Path]:
    """
    Given inputs (files, dirs, globs), return matching file Paths.
    """
    paths: list[Path] = []
    for spec in inputs:
        p = Path(spec)
        if p.is_dir():
            children = sorted(p.iterdir())
            for child in children:
                if child.is_file():
                    paths.append(child)
        elif any(ch in spec for ch in ['*', '?', '[']):
            matches = sorted(glob.glob(spec))
            for match in matches:
                mp = Path(match)
                if mp.is_file():
                    paths.append(mp)
        elif p.is_file():
            paths.append(p)
        else:
            logger.warning(f"No files found for input spec: {spec}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Parse documents using VLM backends (vlm-sglang-engine or vlm-sglang-client)."
    )
    parser.add_argument(
        '-i', '--input', '-p', '--path', nargs='+', required=True,
        help='Files, directories, or glob patterns to parse'
    )
    parser.add_argument(
        '-o', '--output', type=Path, required=True,
        help='Directory to write output files'
    )
    parser.add_argument(
        '-b', '--backend', choices=['vlm-sglang-engine', 'vlm-sglang-client'],
        default='vlm-sglang-engine', help='VLM backend to use'
    )
    parser.add_argument(
        '-u', '--server-url', default="http://127.0.0.1:30000",
        help='Server URL for client backend (e.g., http://127.0.0.1:30000)'
    )
    parser.add_argument(
        '-n', '--num-workers', type=int, default=1,
        help='Number of worker processes for sglang-client backend'
    )
    parser.add_argument(
        '--no-layout-box', dest='f_draw_layout_bbox', action='store_false',
        help='Disable drawing of layout bounding boxes'
    )
    parser.add_argument(
        '--span-box', dest='f_draw_span_bbox', action='store_true',
        help='Enable drawing of span bounding boxes'
    )
    parser.add_argument(
        '--no-md', dest='f_dump_md', action='store_false',
        help='Disable dumping of Markdown output'
    )
    parser.add_argument(
        '--no-middle-json', dest='f_dump_middle_json', action='store_false',
        help='Disable dumping of middle JSON output'
    )
    parser.add_argument(
        '--no-model-output', dest='f_dump_model_output', action='store_false',
        help='Disable dumping of model inference output'
    )
    parser.add_argument(
        '--no-orig-pdf', dest='f_dump_orig_pdf', action='store_false',
        help='Disable copying of original PDF to output'
    )
    parser.add_argument(
        '--no-content-list', dest='f_dump_content_list', action='store_false',
        help='Disable dumping of content list JSON'
    )
    args = parser.parse_args()

    if not args.input:
        logger.error("At least one input spec is required")
        return

    if args.server_url:
        args.backend = "vlm-sglang-client"
    if args.backend.startswith("vlm-"):
        args.backend = args.backend.replace("vlm-", "")

    pdf_paths = expand_inputs(args.input)
    if not pdf_paths:
        logger.error("No valid input files to process. Exiting.")
        return

    try:
        args.output.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory {args.output}: {e}")
        return

    # Parallel processing for sglang-client using processes
    if args.backend == 'sglang-client' and args.num_workers > 1:
        logger.info(f"Running with {args.num_workers} worker processes")
        tasks = [
            (
                args.output,
                [path],
                args.backend,
                args.server_url,
                args.f_draw_layout_bbox,
                args.f_draw_span_bbox,
                args.f_dump_md,
                args.f_dump_middle_json,
                args.f_dump_model_output,
                args.f_dump_orig_pdf,
                args.f_dump_content_list
            )
            for path in pdf_paths
        ]
        with Pool(processes=args.num_workers) as pool:
            pool.starmap(do_parse, tasks)
    else:
        do_parse(
            output_dir=args.output,
            pdf_paths=pdf_paths,
            backend=args.backend,
            server_url=args.server_url,
            f_draw_layout_bbox=args.f_draw_layout_bbox,
            f_draw_span_bbox=args.f_draw_span_bbox,
            f_dump_md=args.f_dump_md,
            f_dump_middle_json=args.f_dump_middle_json,
            f_dump_model_output=args.f_dump_model_output,
            f_dump_orig_pdf=args.f_dump_orig_pdf,
            f_dump_content_list=args.f_dump_content_list,
        )


if __name__ == '__main__':
    main()
