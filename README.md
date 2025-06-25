# MinerU CLI

**MinerU CLI** is a command-line wrapper around the [OpenDataLab MinerU](https://github.com/opendatalab/MinerU) library. It provides:

* **Robust error handling** to isolate and report failures per document without halting the entire batch.
* **PDF preprocessing via PyMuPDF**, avoiding pdfmium decoding/encoding errors and improving stability.

---

## 🚀 Features

* **Flexible Input**: Accepts single files, directories, or glob patterns (`*.pdf`).
* **Multiple Backends**: Choose between local (`vlm-sglang-engine`) or client/server (`vlm-sglang-client`) modes.
* **Rich Outputs**:

  * Original PDF copy
  * Markdown summary
  * Content list JSON
  * Middle-layer JSON
  * Raw model output text
* **Visualizations**:

  * Layout bounding boxes
  * Span bounding boxes
* **Configurable** via command-line flags for fine control over what gets generated.
* **Isolated Processing**: Each document is processed independently with detailed logging.

---

## 📦 Installation

MinerU CLI is available via `pip`. From the project root:

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/DotIN13/mineru-cli.git
```

This will install the `mineru-cli` executable in your environment.

---

## ⚙️ Usage

### Basic Command

```bash
mineru-cli \
  --input path/to/doc.pdf \
  --output ./out_dir
```

Users can choose between two processing modes:

1. **Local processing** with the built-in engine:

   ```bash
   mineru-cli --backend vlm-sglang-engine --input file.pdf --output out_dir
   ```

2. **Client/server mode** for improved performance:

   * First, start the server:

     ```bash
     export MINERU_MODEL_SOURCE=modelscope # modelscope, huggingface or local
     mineru-sglang-server
     ```

   * Then run:

     ```bash
     mineru-cli --backend vlm-sglang-client \
       --server-url http://127.0.0.1:30000 \
       --input docs/*.pdf \
       --output results
     ```

### Options

| Flag                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `-i, --input`       | Files, directories, or glob patterns to parse (required)     |
| `-o, --output`      | Directory to write output files (required)                   |
| `-b, --backend`     | `vlm-sglang-engine` or `vlm-sglang-client` (default: engine) |
| `-u, --server-url`  | URL for client backend mode                                  |
| `--no-layout-box`   | Disable layout bounding box visualizations                   |
| `--span-box`        | Enable span bounding box visualizations                      |
| `--no-md`           | Do not dump Markdown output                                  |
| `--no-middle-json`  | Do not dump intermediate JSON                                |
| `--no-model-output` | Do not dump raw model output                                 |
| `--no-orig-pdf`     | Do not copy original PDF to output                           |
| `--no-content-list` | Do not dump content list JSON                                |

---

## 🛠️ Development

Clone the repo and install:

```bash
git clone https://github.com/DotIN13/mineru-cli.git
cd mineru
pip install -e .
```

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for:

* Feature requests
* Bug reports
* Improvements to CLI or documentation

Be sure to follow existing code style and add tests where appropriate.

---

## 📄 License

This project is licensed under the [AGPL-3.0 License](LICENSE).
