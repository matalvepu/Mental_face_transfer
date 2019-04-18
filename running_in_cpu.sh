#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:30:08 2019

@author: mhasan8
"""


#!/bin/bash
#SBATCH -o  /scratch/mhasan8/output/ets_normal_output%j.txt -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=4gb
#SBATCH -J kamrul
#SBATCH -p standard
#SBATCH -a 0-10

module load anaconda3/5.3.0b
module load git
python running_different_configs.py 
