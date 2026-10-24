# Issue #17: How we run notebooks on Kaggle from the command line

## Question

How do we push a private notebook, pick the hardware, attach inputs, read the log and pull the output, all from the terminal? And what are the limits?

## What we tried

A smoke notebook (`notebooks/smoke/`) that imports our code, checks the frozen folds against the competition data and prints the environment. It fits nothing. Run three times: twice on CPU, once on a T4 pair.

## Findings

**Limits** (Kaggle docs, plus `kaggle quota` for ours)

| | |
|---|---|
| Session length | 12 h on CPU and GPU, 9 h on TPU |
| Scoring run for the competition | 9 h, internet off |
| Saved output | 20 GB in `/kaggle/working` |
| CPU machine | 4 cores, 30 GB RAM |
| GPU machine | 2 T4s (15 GB each), 4 cores, 29 GB RAM |
| Our quota | 30 GPU hours and 20 TPU hours a week, reset 31 Oct |

Kaggle doesn't document how many sessions one account can run at once. We find out the first time we launch runs in parallel.

**The notebook settings file.** Each notebook folder holds the code and a `kernel-metadata.json`. The fields that matter to us:

- `id`: `nabilshah/<slug>`.
- `kernel_type`: `script` for a `.py` file, `notebook` for `.ipynb`.
- `is_private`: `true`. It is also the default.
- `enable_internet`: `false`, the same as the scoring run.
- `competition_sources`: `["name-that-llm"]`.
- `dataset_sources`: `owner/slug`, for example our code and the public pools.
- `model_sources`: `owner/model/framework/variation/version`, for example `qwen-lm/qwen-3/Transformers/14b-base/1`.
- `kernel_sources`: another notebook's output as an input. This is how a training notebook hands its weights to the next notebook.

**Hardware** is picked at push time with `--accelerator NvidiaTeslaT4` (two T4s) or `TpuV5E8`. The P100 is retired. `-t <seconds>` caps the run time.

**Where inputs land** (seen in the log):

- Competition data: `/kaggle/input/competitions/name-that-llm/`
- Datasets: `/kaggle/input/datasets/<owner>/<slug>/`

**Our code** goes up as the private dataset `nabilshah/nthl-src` with `scripts/push_src.sh`, versioned by commit hash. Kaggle unpacks it flat, so a notebook copies it to `/tmp/pkg/nthl` before importing. It stays private because it is our own work (see AGENTS.md).

**The environment** on the CPU image: Python 3.13, numpy 2.1, pandas 2.3, scikit-learn 1.6, pyarrow 23, torch 2.11, transformers 5.16, peft 0.20, accelerate 1.14, LightGBM 4.6. No bitsandbytes. Locally we run pandas 3.1, so our code has to work on both.

**The GPU run.** `--accelerator NvidiaTeslaT4` was enough on its own, with `enable_gpu` left false in the settings file. We got two Tesla T4s and torch 2.11 with CUDA 12.8. The run took 45 seconds, about 0.01 GPU hours. bitsandbytes is missing on the GPU image too, so loading a model in 4 or 8 bit needs an offline wheel. Passed on to #22.

**Traps we hit**

- `kaggle datasets status` answered 403 for a minute after the dataset was created, and the file list lagged behind the upload. Both fixed themselves. `push_src.sh` now checks with `datasets list --mine` instead.
- The first smoke run failed because the code dataset unpacks flat. Fixed by copying it into a folder named `nthl`.

## Recommendation

Run every experiment as a private script notebook pushed from the repo, with internet off so it behaves like the scoring run. Upload the code first if it changed, then push, follow the log and pull the output into `outputs/<slug>/`. Record the code version and the notebook version in the experiment log.

```bash
scripts/push_src.sh                                                # upload src/nthl if it changed
.venv/bin/kaggle kernels push -p notebooks/<name>                  # CPU run
.venv/bin/kaggle kernels push -p notebooks/<name> --accelerator NvidiaTeslaT4 -t 3600   # two T4s, capped at an hour
.venv/bin/kaggle kernels status nabilshah/<slug>                   # QUEUED, RUNNING, COMPLETE or ERROR
.venv/bin/kaggle kernels logs nabilshah/<slug> -f                  # follow the log live
.venv/bin/kaggle kernels output nabilshah/<slug> -p outputs/<slug> # pull the output files and the log
.venv/bin/kaggle quota                                             # GPU and TPU hours left
```

Pushing a notebook is not a submission. Nothing goes to the competition without Nabil's go-ahead.
