# AdaMCL: Adaptive Multi-View Fusion Contrastive Learning for Collaborative Filtering

This is the official PyTorch implementation of AdaMCL:

## Overview

We propose a novel adaptive multi-view fusion contrastive learning framework, named (**AdaMCL**), for graph collaborative filtering.

<div  align="center"> 
<img src="https://github.com/PasaLab/AdaMCL/blob/main/framework.jpg" style="width: 75%"/>
</div>

## Requirements

```
recbole==1.0.0
python==3.7.7
pytorch==1.7.1
```

## Quick Start

```bash
python main.py --dataset ml-1m
```

You can replace `ml-1m` to `yelp`, `amazon-books`, `gowalla-merged` or `alibaba` to reproduce the results reported in our paper.


