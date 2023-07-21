#!/bin/bash

python main.py --gpu=0 --epoch=6 --embed_dim=100

python main.py --gpu=1 --epoch=6 --embed_dim=200

python main.py --gpu=2 --epoch=6 --embed_dim=300