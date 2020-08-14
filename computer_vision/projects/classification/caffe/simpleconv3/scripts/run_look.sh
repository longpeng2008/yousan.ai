#!/usr/bin/env bash
# Copyright 2019 longpeng2008. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# If you find any problem,please contact us
#
#     longpeng2008to2012@gmail.com 
#
# or create issues
./geneaccrefine
./genelossrefine
python show_loss.py
python show_acc.py
