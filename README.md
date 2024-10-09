# LiNeS

This is a source code to reproduce the experiments for "[LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging](https://arxiv.org/abs/todo)".

Our paper proposes a post-training model edition method to mitigate catastrophic forgetting, with applications for improving model merging methods.
This repo contains the following experiments:
1) applying LiNeS on the fine-tuned residual, improving fine-tuned model's performance on control tasks while preserving performance on the fine-tuned (target) task.
2) applying LiNeS for enhancing multi-task merging, improves the performance over multiple baseline merging methods.

<!-- ![](figures/illustration.png) -->

## Dependencies

To run the code, please install all its dependencies:
```sh
conda env create
conda activate lines
```

## Checkpoints
We use the checkpoints from [this link](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae?usp=sharing), where you can download the checkpoints by running the following script:
```sh
# model options --model {ViT-B-32,ViT-L-14} 
# use python download_checkpoints.py --help for more information
python download_checkpoints.py --model='ViT-B-32' --kind=checkpoints
```

The script downloads *all* the checkpoints for one model corresponding to 40 files. If you encounter any issues, please refer to the [gdown documentation](https://github.com/wkentaro/gdown?tab=readme-ov-file#faq). A common issue is that the download quota is exceeded, in which case you can download the files manually from the [Google Drive folder](https://drive.google.com/drive/folders/15ParSng4d5xSdaWdBFsg1617zPXT8Dae?usp=sharing) or modify your local cookies file as described in the gdown documentation.

## Datasets
Most datasets being used should be downloaded automatically with torchvision or huggingface. For the datasets requiring manual preparation, please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1). Depending on the torchvision version, some issues might arise when downloading specific datasets like [here](https://github.com/basveeling/pcam/issues/4) or [here](https://github.com/pytorch/vision/issues/5662). In this case, using a different torchvision version might solve the issue. 

## Evaluation
Evaluation is performed with Hydra, please modify `model_location` and `data_location` in `config/config.yaml` before evaluation. Note that you can set different number of tasks by setting `num_tasks`. Then, the first `num_tasks` are going to be selected from the list defined in `src/utils/variables_and_paths.py`.

### 1) Editing the fine-tuned checkpoint

We provide in `LiNeS_example.ipynb` an example for applying LiNeS to edit the fine-tuned checkpoint, such that it reduces the performance loss on the control datasets while preserving fine-tuned accuracy.

Alternatively, you can run the following scripts:

```bash
# Evaluate the performance of the pre-trained model on the first 8 tasks
python main.py model=ViT-B-32 num_tasks=8 method="zero-shot"

# Evaluate the performance of fine-tuned model on the first 8 tasks; task_index=0 indicates fine-tuned on the 0-th task
python main.py model=ViT-B-32 num_tasks=8 method="single_task" method.task_index=0

# Evaluate the performance of fine-tuned model (edited with LiNeS) on the first 8 tasks; task_index=0 indicates fine-tuned on the 0-th task
python main.py model=ViT-B-32 num_tasks=8 method="single_task" method.apply_lines=True method.task_index=0
```

### 2) Improving multi-task merging baselines

We apply LiNeS to enhance the baseline multi-task merging methods by scaling the multi-task vector.

The following scirpts demonstrate the usage:
```bash
# Evaluate with Task Arithmetic baseline
python main.py model=ViT-B-32 num_tasks=8 method="sum"

# Evaluate with Task Arithmetic baseline; enhanced with LiNeS
python main.py model=ViT-B-32 num_tasks=8 method="sum" num_tasks=8 method.apply_lines=True

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
* Enhancing with LiNeS does not increase the hyper-parameter tuning costs compared to baseline methods.
* You can select model in [ViT-B-32, ViT-L-14] and num_tasks in [8, 14, 20] to test different settings in the paper.
* For consensus merging, you need to construct TALL-masks in advance, details are in [this link](https://github.com/nik-dim/tall_masks).


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
@article{,
  title={LiNeS: Post-training Layer Scaling Prevents Forgetting and Enhances Model Merging},
  author={},
  journal={},
  year={2024}
}
```