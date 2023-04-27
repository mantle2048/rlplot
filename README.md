<div align="center">
  <img width="300px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/rlplot.png"></a>
</div>

 <br>


`rlplot` is an easy to use and highly encapsulated RL plot library (including basic error bar lineplot and a wrapper to [rliable](https://github.com/google-research/rliable)).

<br> 
 

 <div align="center">
  <img width="700px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/dataflow.png"></a>
</div>

<br>

## Installation

To install `rlplot`, please run:

```bash
git clone https://github.com/mantle2048/rlplot
cd rlplot
pip install -e .
```

### Requirements

- hydra-core >= 1.3.0
- arch == 5.3.0
- scipy >= 1.7.0
- numpy >= 0.9.0
- absl-py >= 1.16.4
- seaborn >= 0.11.2

After installation, open your python console and type

```python
import rlplot
print(rlplot.__version__)
```

If no error occurs, you have successfully installed `rlplot`.

## Demo

Here, we provide an illustration on how to utilize this library for analysis and plot of your RL experiments.

### 0. Preliminary

Assum that you have four algorithms `[D, C, B, A]` and four tasks `[HalfCheetah-v4, Ant-v4, Walker2d-v4, Humanoid-v4]`.

The experimental logs must be put in `exps` folder with the following file tree:

``` bash
└── ${algo}_${task}(i.e. A_Ant-v4) # exp_name
    └── ${timestamp}_${algo}_${task}_${seed}(i.e. 2023-01-14_02-36-30_A_Ant-v4_101) # run_name, "timestamp" is optional
        └── progress.csv # experiment log, including metrics like "AverageEvalReward", "TotalEnvInteracts", "Epoch", etc
```

### 1. The create of diagnosis

Convert all `progress.csv` files in `exps` to yaml file and store them in `diagnosis` folder.

For instance, you can run the following command:

```python
python main.py \
    type=create_diagnosis \
    n_epoch=200 \
    epoch_len=5000 \
    milestone_gap=100000 \
    metric='AverageEvalReward' \
    algos=['D','C','B','A'] \
    tasks=['HalfCheetah-v4','Ant-v4','Walker2d-v4','Humanoid-v4']
```

Alternatively, you can manually modify `cfgs/config.yaml` and `cfgs/type/create_diagnosis.yaml` and then just run `make diag`.

### 2. Plot figures

#### 2.1. Metric curve

```python
python main.py \
    type=metric_curve \
    n_epoch=200 \
    epoch_len=5000 \
    metric='AverageEvalReward' \
    aggregate_name='IQM' \
    smooth_size=1 \
    algos=['D','C','B','A'] \
    tasks=['HalfCheetah-v4','Ant-v4','Walker2d-v4','Humanoid-v4']
```
Across all tasks           |      HalfCheetah-v4           |      Ant-v4           |      Walker2d-v4           |      Humanoid-v4
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
<img width="150px"  src="https://github.com/mantle2048/my_assets/blob/master/metric_curve.png"> | <img width="150px"  src="https://github.com/mantle2048/my_assets/blob/master/metric_curve_halfcheetah-v4.png"> | <img width="150px"  src="https://github.com/mantle2048/my_assets/blob/master/metric_curve_ant-v4.png"> | <img width="150px"  src="https://github.com/mantle2048/my_assets/blob/master/metric_curve_walker2d-v4.png"> | <img width="150px"  src="https://github.com/mantle2048/my_assets/blob/master/metric_curve_humanoid-v4.png">


#### 2.2. Metric value

```python
python main.py \
    type=metric_value \
    milestone='1m' \
    aggregate_names=['Mean','IQM'] \
    algos=['D','C','B','A'] \
```

<div align="center">
  <img width="400px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/metric_value.png"></a>
</div>

#### 2.3. Performance profiles

```python
python main.py \
    type=performance_profiles \
    milestones=['300k','500k','1m'] \
    algos=['D','C','B','A']
```

<div align="center">
  <img width="600px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/performance_profiles.png"></a>
</div>

#### 2.4. Probability of improvement

```python
python main.py \
    type=probability_of_improvement \
    milestone='1m' \
    algos=['D','C','B','A']
```

<div align="center">
  <img width="400px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/probability_of_improvement.png"></a>
</div>


#### 2.5. Sample efficiency curve

```python
python main.py \
    type=sample_efficiency_curve \
    steps=[1,10,25,50,75,100,125,150,175,200] \
    algos=['D','C','B','A']
```

<div align="center">
  <img width="300px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/sample_efficiency_curve.png"></a>
</div>

#### 2.6. Overall ranks

```python
python main.py \
    type=overall_ranks \
    milestones=['300k','500k','1m'] \
    algos=['D','C','B','A']
```

<div align="center">
  <img width="600px" height="auto" src="https://github.com/mantle2048/my_assets/blob/master/overall_ranks.png"></a>
</div>

#### 2.7. One command for all plot

For a "lazy" person :), you can modify the yaml config files in `cfgs` for your case , then just run `make all`.

-----------
## Video
[![asciicast](https://asciinema.org/a/K9rOCDVC0ULSaUpkvfA1FhNnw.svg)](https://asciinema.org/a/K9rOCDVC0ULSaUpkvfA1FhNnw)

# Trivia

The main code structure in `rlplot` is very simple, and most of it is inherited from `rliable`. You can completely modify your favorite painting style in `plot_utils.py`. All credit goes to the author of `rliable`. For more details about the specific meaning of above plots, please refer to the original library [`rliable`](https://github.com/google-research/rliable).

