#!/usr/bin/env bash
set -e
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
B=https://www.research-collection.ethz.ch/server/api/core/bitstreams
mkdir -p /data/shared/euroc_dl && cd /data/shared/euroc_dl
for spec in vicon_room1:02ecda9a-298f-498b-970c-b7c44334d880:V1_01_easy,V1_03_difficult vicon_room2:ea12bc01-3677-4b4c-853d-87c7870b8c44:V2_01_easy,V2_03_difficult; do
  IFS=: read -r room id seqs <<<"$spec"
  curl -s -A "$UA" -L -C - --retry 20 --retry-delay 30 -o $room.zip "$B/$id/content"
  echo "$room downloaded $(stat -c %s $room.zip) $(date +%T)"
  for s in ${seqs//,/ }; do
    short=$(echo $s | sed -E 's/^V([12])_0([0-9])_.*/V\1\2/')
    d=/data/shared/$short; [ -e $d ] && { echo "$d exists"; continue; }
    unzip -q -o $room.zip "$room/$s/$s.zip" -d . && mkdir -p $d && unzip -q "$room/$s/$s.zip" -d $d
    ln -sfn $d /data/tkocher/datasets/euroc/$s
    echo "$s -> $d $(ls $d/mav0/cam0/data | wc -l) frames $(du -sh $d | cut -f1)"
  done
  rm -rf $room $room.zip
done
cd .. && rmdir euroc_dl; echo "=== vicon download complete $(date +%T) ==="
