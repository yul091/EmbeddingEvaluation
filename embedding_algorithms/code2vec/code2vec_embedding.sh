#!/bin/bash

#python main.py --device=3 --epochs=2 --embed_dim=100

python main.py --batch=64 --device=4 --epochs=2 --embed_dim=200

python main.py --batch=64 --device=5 --epochs=2 --embed_dim=300