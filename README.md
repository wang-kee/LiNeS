# LiNeS

This is a source code to reproduce the experiments for "[LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging](https://arxiv.org/abs/2410.17146)" by Ke Wang*, Nikolaos Dimitriadis*, Alessandro Favero, Guillermo Ortiz-Jimenez, Francois Fleuret, and Pascal Frossard. 

Our paper proposes a post-training model editing method to mitigate catastrophic forgetting, with applications for improving model merging methods.
This repo contains the following experiments:
1) applying LiNeS on the fine-tuned residual, improving fine-tuned model's performance on control tasks while preserving performance on the fine-tuned (target) task.
2) applying LiNeS for enhancing multi-task merging, improving the performance over multiple baseline merging methods.

This repo is heavily based on the repo for [TALL-Masks](https://github.com/nik-dim/tall_masks).

<!-- ![](figures/illustration.png) -->

## Dependencies

To run the code, please install all its dependencies:
```sh
conda env create
conda activate lines
```

## Checkpoints

The checkpoints can be downloaded from the HuggingFace repo [`nik-dim/tall_masks`](https://huggingface.co/nik-dim/tall_masks). See the [`snapshot_download documentation`](https://huggingface.co/docs/huggingface_hub/v0.26.0/en/package_reference/file_download#huggingface_hub.snapshot_download) for more details.

```sh
from huggingface_hub import snapshot_download

# download the ViT-B-32 checkpoints including backbone, classification heads and tall masks
snapshot_download(repo_id="nik-dim/tall_masks", allow_patterns="*32*")

# download the ViT-B-16 checkpoints including backbone, classification heads and tall masks
snapshot_download(repo_id="nik-dim/tall_masks", allow_patterns="*16*")

# download the ViT-L-14 checkpoints including backbone, classification heads and tall masks
snapshot_download(repo_id="nik-dim/tall_masks", allow_patterns="*14*")

# download everything
snapshot_download(repo_id="nik-dim/tall_masks")
```

## Datasets
Most datasets being used should be downloaded automatically with torchvision or huggingface. For the datasets requiring manual preparation, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1). Depending on the torchvision version, some issues might arise when downloading specific datasets like [here](https://github.com/basveeling/pcam/issues/4) or [here](https://github.com/pytorch/vision/issues/5662). In this case, using a different torchvision version might solve the issue. 

## Evaluation
Evaluation is performed with Hydra, please modify `model_location` and `data_location` in `config/config.yaml` before evaluation. Note that you can set different number of tasks by setting `num_tasks`. Then, the first `num_tasks` are going to be selected from the list defined in `src/utils/variables_and_paths.py`, which you can modify as well.

### 1) Editing the fine-tuned checkpoint

We provide in `LiNeS_example.ipynb` an example for applying LiNeS to edit the fine-tuned checkpoint, such that it reduces the performance loss on the control datasets while preserving fine-tuned accuracy.

Alternatively, you can run the following scripts:

```bash
# Evaluate the zero-shot performance of the pre-trained model on the first 8 tasks
python main.py model=ViT-B-32 num_tasks=8 method="zeroshot"

# Evaluate the performance of fine-tuned model on the first 8 tasks; 
# task_index=0 indicates fine-tuned on the 0-th task
python main.py model=ViT-B-32 num_tasks=8 method="single_task" method.task_index=0

# Evaluate the performance of fine-tuned model (edited with LiNeS) on the first 8 tasks; 
# task_index=0 indicates fine-tuned on the 0-th task
python main.py model=ViT-B-32 num_tasks=8 method="single_task" method.apply_lines=True method.task_index=0
```

The target and control tasks accuracy are separately reported when evaluating the fine-tuned checkpoints. Note that you can set different vaxlues to `method.tradeoff_target_weight` (set by default to 2) to select varying importance to target accuracy (for the trade-off between target and control task accuracy) when selecting the best hyper-parameter for LiNeS for evaluation on test set.

### 2) Improving multi-task merging baselines

We apply LiNeS to enhance the baseline multi-task merging methods by scaling the multi-task vector.

The following scirpts demonstrate the usage:
```bash
# Evaluate with Task Arithmetic baseline
python main.py model=ViT-B-32 num_tasks=8 method="sum"

# Evaluate with Task Arithmetic baseline; enhanced with LiNeS
python main.py model=ViT-B-32 num_tasks=8 method="sum" method.apply_lines=True

# Evaluate with Ties-merging baseline
python main.py model=ViT-B-32 num_tasks=8 method="ties" method.k=20

# Evaluate with Ties-merging baseline; enhanced with LiNeS
python main.py model=ViT-B-32 num_tasks=8 method="ties" method.k=20 method.apply_lines=True

# Evaluate with Consensus merging baseline
python main.py model=ViT-B-32 num_tasks=8 method="consensus" method.prun_thre_k=2

# Evaluate with Consensus merging baseline; enhanced with LiNeS
python main.py model=ViT-B-32 num_tasks=8 method="consensus" method.prun_thre_k=2 method.apply_lines=True
```

Notes:
* Enhancing with LiNeS maintains the same hyper-parameter tuning costs compared to baseline methods.
* You can select model in [ViT-B-32, ViT-L-14] and num_tasks in [8, 14, 20] to test different settings in the paper.
* For consensus merging, you need to construct TALL-masks in advance, details in [this link](https://github.com/nik-dim/tall_masks).


## Other usage:

``` sh
# Finetune on 2 GPUs
python finetune.py --model=ViT-B-32 --world-size=2 

# Evaluate pre-trained model on single task
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate fine-tuned model on single task
python eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard

```

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@article{wang2024lines,
author = {
    Ke Wang,
    Nikolaos Dimitriadis,
    Alessandro Favero,
    Guillermo Ortiz-Jimenez,
    Fran\c{c}ois Fleuret,
    Pascal Frossard},
journal = {arXiv},
title = {{LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging}},
year = {2024}
}

```
