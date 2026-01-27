# RubikSQL

RubikSQL is a general-purpose instance-optimized NL2SQL solution built on top of [AgentHeaven](https://github.com/RubikSQL/AgentHeaven).



https://github.com/user-attachments/assets/46eec655-8c00-4605-85c6-f210fa4c1ff3


## Update Log

- [2026-01] Initial release of **RubikBench v0.9** database on huggingface: [RubikBench](https://huggingface.co/datasets/Magolor/RubikBench).

- [2025-12] [RubikSQL demo](https://github.com/RubikSQL/RubikSQL/blob/main/demo.mp4). Code, Python package, CLI, GUI (web+tauri) and packaged app will be released soon.

- [2025-11] Initial relase of AgentHeaven, the framework used to build RubikSQL: [AgentHeaven](https://github.com/RubikSQL/AgentHeaven), [Documentation (To Be Updated)](https://rubiksql.github.io/AgentHeaven-docs/en/index.html).

---

## Result Highlights

(by August 2025, with `gemini-2.5-flash`)

### BIRD Mini-Dev / Dev

| Method                 | TTS  | Dev EX (%)          |
|------------------------|------|---------------------|
| `gemini-2.5-flash`     | n=1  | 59.4 (Mini-Dev)     |
| CSC-SQL                | n=72 | 71.33               |
| XiYan-SQL              | n=5  | 73.34               |
| Contextual-SQL         | n=32 | 73.50               |
| CHASE-SQL              | n=21 | 74.90               |
| AskData                | n=3  | 73.0 / 75.36        |
| **Rubik (Ours)**       | n=1  | **75.9** (Mini-Dev) |
| **Rubik (Ours)**       | n=8  | **77.3** (Mini-Dev) |

> Mini-Dev is a subset of BIRD Dev; other methods are reported on Dev.

### KaggleDBQA

| Method        | TTS | Test EX (%) |
|---------------|-----|-------------|
| RAT-SQL       | n=1 | 26.8        |
| DIN-SQL       | n=1 | 27.0        |
| ZeroNL2SQL    | n=1 | 44.9        |
| ODIS-Codex    | n=1 | 54.8        |
| **Rubik (Ours)** | n=1 | **54.1** |
| **Rubik (Ours)** | n=8 | **58.9** |

---

## Citation

If you find Rubik or RubikBench useful, please cite:

```bibtex
@misc{chen2025rubiksqllifelonglearningagentic,
      title={Rubik: Bridging the NL2SQL Research-to-Production Gap via Lifelong Learning Agentic Knowledge Base}, 
      author={Zui Chen and Han Li and Xinhao Zhang and Xiaoyu Chen and Chunyin Dong and Yifeng Wang and Xin Cai and Su Zhang and Ziqi Li and Chi Ding and Jinxu Li and Shuai Wang and Dousheng Zhao and Sanhai Gao and Guangyi Liu},
      year={2025},
      eprint={2508.17590},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2508.17590}, 
}
```
