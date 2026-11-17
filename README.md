# DeepHybrid

DeepHybrid adapts [DeepACO](https://arxiv.org/abs/2309.14032) to the **job-shop scheduling problem (JSSP)**. A graph neural network learns which operations are promising to schedule next, and ant colony optimization (ACO) uses those scores to search for shorter schedules.

In a job shop, each job is a sequence of operations. Each operation has a fixed processing time and a designated machine, and each machine handles one operation at a time. The objective is to minimize **makespan**, the time at which the last job finishes.

## How it works

1. **Represent the job shop as a graph.** Each node is an operation with a processing time, a position in its job, and a machine ID. Directed edges connect successive operations within a job, and bidirectional edges connect operations that compete for the same machine.
2. **Learn a scheduling heuristic.** The custom graph inputs feed [RL4CO](https://pypi.org/project/rl4co/)'s non-autoregressive graph neural network (NARGNN) encoder to produce a heatmap of operation-to-operation scores. The model trains on generated job shops with a REINFORCE objective and a terminal reward of negative makespan.
3. **Search with a colony of ants.** At inference, each ant builds a schedule one operation at a time. The environment masks operations whose predecessors are unfinished or whose machines are busy, and advances to the next completion event when no operation can start. Ants sample from the remaining choices using the learned heatmap and pheromone trails. Shorter schedules receive stronger pheromone deposits, and the search keeps the best schedule it finds.

The heatmap gives the search guidance specific to each job shop before any ant builds a schedule. Pheromone updates then use feedback from completed schedules to refine that guidance for the same instance.

## Benchmark results

On the six **10-job, 10-machine** instances below, DeepHybrid cut makespan by **21.9–28.0%** relative to the report's standard ACO baseline and improved on both simple dispatch baselines. The gap to the reported reference ranged from **3.7% to 12.4%**.

| Instance | Reference | Random | First valid | Standard ACO | DeepHybrid | Gap to reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ORB01 | 1,059 | 1,255 | 1,450 | 1,545 | **1,160** | 9.5% |
| FT10 | 930 | 1,255 | 1,265 | 1,405 | **1,045** | 12.4% |
| ABZ5 | 1,234 | 1,485 | 1,370 | 1,775 | **1,280** | 3.7% |
| LA16 | 945 | 1,077 | 1,230 | 1,293 | **1,010** | 6.9% |
| LA17 | 784 | 1,009 | 894 | 1,095 | **814** | 3.8% |
| LA18 | 848 | 909 | 1,128 | 1,252 | **902** | 6.4% |

Makespan is lower-is-better. The gap is `100 × (DeepHybrid − reference) / reference`. Values are from Table 4.1 of the 2026 project report, *DeepACO for Optimization of Job Shop Scheduling Problem*. The report lists ABZ5's reference as best-known and the other five as optima. The comparison covers these six instances.

## Run the notebooks

### Train a model

1. Open [`train.ipynb` in Colab](https://colab.research.google.com/github/pranceraz/DeepHybrid/blob/main/train.ipynb). Run the setup cell to clone the repository, mount Google Drive, and install the graph dependencies.
2. Set `NUM_JOBS`, `NUM_MACHINES`, and the training settings in the configuration cell. The notebook defaults to generated **6 × 6** instances and 30 epochs. Set both dimensions to **10** to train a model for the benchmark sizes shown above. The notebook uses CUDA when available.
3. Run the training cells. Lightning checkpoints, logs, and the final PyTorch `.pt` state dictionary are written to the timestamped run folder under `MyDrive/DeepHybrid/training_runs`.

### Evaluate a checkpoint

1. Open [`_test.ipynb` in Colab](https://colab.research.google.com/github/pranceraz/DeepHybrid/blob/main/_test.ipynb) and run the setup and Drive-mount cells. Replace the example `MODEL_PATHS` with your own `.pt` or Lightning `.ckpt` paths. Model weights are stored separately from the repository.
2. Set `BENCHMARK_KEYWORDS` to an instance or family from JobShopLib. Use `['ft06']` for a 6-machine checkpoint or `['orb01', 'ft10', 'abz5', 'la16', 'la17', 'la18']` for the reported 10-machine instances. The learned machine embedding has a fixed number of entries, so the checkpoint's machine count must match the instance.
3. Run evaluation and plot the makespans and reference gaps. The notebook includes first-valid and random baselines, an optional ACO run with a uniform heuristic, and a separate cell for standalone ACO and OR-Tools CP-SAT. Run just one of the two standalone comparison cells. The CP-SAT cell can return a feasible solution at its time limit, so an objective value from that cell alone does not establish optimality.

Keep `TRUST_CHECKPOINTS` set to `False` for artifacts from untrusted sources. Setting it to `True` allows the notebook to fall back to loading the full serialized checkpoint.

For local Jupyter use, open the notebooks from a clone of this repository, skip the Colab-only clone and Drive-mount commands, and replace the `/content/...` paths with local paths. The notebook setup cells list their dependencies. The optional CP-SAT comparison also requires `ortools`.

## Repository guide

| File | Role |
| --- | --- |
| [`train.ipynb`](train.ipynb) | Configures RL4CO's DeepACO model and trains it on generated instances. |
| [`_test.ipynb`](_test.ipynb) | Loads checkpoints, evaluates published benchmark instances, and plots comparisons. |
| [`generator.py`](generator.py) | Generates batched instances and operation features. |
| [`init_embedding.py`](init_embedding.py) | Builds operation embeddings and graph connectivity for NARGNN. |
| [`my_env.py`](my_env.py) | Implements the operation-selection environment used by both notebooks. |
| [`env.py`](env.py) | Alternate operation-selection environment. |
| [`aco_class.py`](aco_class.py) | Implements batched ant sampling, pheromone updates, and best-schedule tracking. |

Developed by Arnav Yadavilli and Siripragada Nischala, advised by Dr. Divya Shrivastava. Released under the [MIT license](LICENSE).
