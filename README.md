# MACA-A-Multi-Agent-Cognitive-Adaptation-Framework
## Required Files to Run `maca.py`

The project is centered around `maca.py`, which serves as the main entry point. Make sure the following files are present when cloning or deploying the repository:

| File / Directory | Purpose |
| --- | --- |
| `maca.py` | Main executable that launches the full multi-modal, multi-agent system. |
| `integration_wrapper.py` | Hooks the GPT-4 multi-agent pipeline and GEMMAS metrics into `maca.py`. |
| `paper_reward_function.py` | Implements the R* reward function described in the paper. |
| `llm_agents.py` | Defines the Planner, Critic, and Executor agents. |
| `sequential_collaboration.py` | Orchestrates the Planner → Critic → Executor workflow. |
| `gemmas_framework.py` | Computes IDS and UPR collaboration metrics. |
| `approach/ResEmoteNet.py` | ResEmoteNet architecture used for emotion recognition. |
| `fer2013_model.pth` | Pretrained ResEmoteNet weights (306 MB). Store at project root or update the path in `maca.py`. |
| `requirements.txt` | Python package list needed to run `maca.py`. |
| `data/` | Created automatically at runtime for CSV logs; include an empty folder if you want it versioned. |

> **Important:** `fer2013_model.pth` is large and typically tracked via Git LFS or shared through an external download link. Update the README with instructions if you host it elsewhere.

With these files in place, you can activate your environment, install the requirements, and launch the system:

pip install -r requirements.txt
python maca.py`maca.py` will automatically detect your OpenAI API key (`OPENAI_API_KEY`) and enable the GPT-4 agent pipeline if available. Otherwise, it falls back to the simulation mode.
