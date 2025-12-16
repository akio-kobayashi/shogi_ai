#!/bin/sh
# Copy CSA files from Gogle drive **efficiently** with rclone.
# CSA files should be copied to your local machine for faster processing.

SRC=~/GoogleDrive/floodgate/csa_raw/2025
DST=./csa/
rclone copy gdrive:${SRC} ${DST} \
       --transfers 16 \
       --checkers 16 \
       --fast-list \
       --drive-chunk-size 128M \
       --progress
