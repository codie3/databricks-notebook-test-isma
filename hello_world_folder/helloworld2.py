# Databricks notebook source
# Python Script       : hello_World.py

import random
import sys
import os

dbutils.widgets.text("dt", "")
dbutils.widgets.text("anotherparam", "")

# COMMAND ----------




r = random.choice(range(0,100))

dt = dbutils.widgets.get('dt')
print(f'hello world argument1:, {r}, {dt}')

anotherparam = dbutils.widgets.get('anotherparam')
print(f'hello world argument2, {r}, {anotherparam}')

data_sharing_prefix = os.getenv('DATA_SHARING_PREFIX')
print(f"data_sharing_prefix: {data_sharing_prefix}")


