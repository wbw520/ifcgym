# Third-party components and data

IfcGym and IfcEdit are released under the Apache License 2.0 (see `LICENSE`). The repository and the datasets it refers to
contain or depend on the following third-party work.

## Code

| Component | Location | Licence | Notes |
|---|---|---|---|
| IfcOpenShell 0.8.5 | dependency | LGPL-3.0 | loaded in the sandbox worker; not vendored |
| MCP4IFC (`ifc-bonsai-mcp` 0.1.0, Show2Instruct, main @ 6215493) | `ifcedit/frameworks/mcp4ifc_vendor/` | MIT (Copyright (c) 2025 Bharathi Kannan N.) | port of the authors' tool server to plain IfcOpenShell; the `LICENSE` file is kept verbatim, see the folder's `README.md` for the file-by-file provenance |
| IFC-Agent (Gao, Hu, Chai, Weng, Li, 2026) | `ifcedit/frameworks/ifcagent*.py` | our re-implementation from the paper | no code of the authors is included |
| BIM-Edit evaluator (Nithyanantham et al., 2026) | `ifcedit/bimedit_eval.py`, `ifcedit/bimedit_calib.py` | our re-implementation from the paper | calibrated on the authors' released per-task scores |
| BIBIMBAP evaluation code (Kujat et al., 2026) | expected under `data/bibimbap/code` | the authors' licence | not vendored; `ifcedit/run_bibimbap.py` calls the authors' per-task tests |
| verl 0.9.0 | training dependency | Apache-2.0 | `ifcedit/train/` holds our SFT / GRPO entry points and agent loop |
| Qwen3.5 (Alibaba) | model weights, not included | Apache-2.0 | student backbone |

## Data referred to by the code (see `docs/DATA.md`)

| Dataset | Licence | Use |
|---|---|---|
| ifc-bench (Hellin, 2025; Hugging Face `sylvainHellin/ifc-bench`) | CC BY 4.0 for the collection; each project keeps its own licence file (MIT, CC BY 4.0, GPL-3.0 and buildingSMART sample-file terms occur) | source of the 13 real projects of IfcEditBench and of the development projects |
| CubiCasa5K (Kalervo et al., 2019) | CC BY-NC 4.0 | floor plans from which the synthetic training buildings are generated; everything derived from them (synthetic buildings, generated tasks, teacher episodes, RL pool) is therefore released under CC BY-NC 4.0 |
| BIM-Edit (Nithyanantham et al., 2026) | CC BY 4.0 | external benchmark; downloaded separately |
| BIBIMBAP (Kujat et al., 2026; Hugging Face `clemenskujat/bibimbap-data`) | CC BY 4.0 | external benchmark; downloaded separately |
| Teacher episodes | produced with Kimi K3 (Moonshot AI) in thinking mode | distributed with the synthetic dataset under CC BY-NC 4.0; subject to the model provider's terms of use |
