# Third-party components

DocuQA_v2 is a local document analysis application. Original application code is separate
from third-party packages. `uv.lock` records exact dependency versions; uv
installs their upstream distributions and associated license files.

Core upstream license references:

| Component | License | Upstream |
|---|---|---|
| word2pdf | MIT | https://github.com/surenjanath/Word2PDF |
| ReportLab | BSD | https://www.reportlab.com/ |
| Streamlit | Apache-2.0 | https://github.com/streamlit/streamlit |
| LangChain / langchain-ollama | MIT | https://github.com/langchain-ai/langchain |
| LangGraph | MIT | https://github.com/langchain-ai/langgraph |
| rank-bm25 | Apache-2.0 | https://github.com/dorianbrown/rank_bm25 |
| Chroma | Apache-2.0 | https://github.com/chroma-core/chroma |
| Nomic Embed Text v1.5 weights (downloaded separately) | Apache-2.0 | https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 |
| pypdf | BSD-3-Clause | https://github.com/py-pdf/pypdf |
| python-docx | MIT | https://github.com/python-openxml/python-docx |
| pypdfium2 / PDFium | Multiple permissive licenses; see bundled notices | https://github.com/pypdfium2-team/pypdfium2 |
| Ollama runtime (installed separately) | MIT, plus bundled dependencies | https://github.com/ollama/ollama |
| Qwen3-4B weights (downloaded separately) | Apache-2.0 | https://huggingface.co/Qwen/Qwen3-4B |

This source project does not redistribute model weights or an Ollama binary.
If making a bundled installer later, collect the complete license texts and
applicable NOTICE files for **all** shipped packages, model weights, Python,
PDFium and GPU libraries. These links are not a substitute for those notices.
