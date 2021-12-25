#!/bin/bash
#
# desc: mv snapshot 脚本到其他的的磁盘



# --------- 定义变量 START -----------
# 1. 要存到的文件夹
TARGET_DIR="/tmp0/data"
# 2. scylladb 的数据文件夹
SCYLLA_DATA_DIR="/var/lib/scylla"
# 3. snapshot 的名称
SNAPSHOT_NAME="1638870422768"
# --------- 定义变量 END -----------


# --------- 衍生变量 START -----------
TARGET_PATH="${TARGET_DIR}/${SNAPSHOT_NAME}"
# --------- 衍生变量 END-----------


for i in `find ${SCYLLA_DATA_DIR} -name ${SNAPSHOT_NAME}`
do
  SNAP_DIR_PATH="${TARGET_PATH}${i}"
  mkdir -p "${SNAP_DIR_PATH}"
  rm -rf "${SNAP_DIR_PATH}/*"
  echo "cp -r ${i}/* ${SNAP_DIR_PATH}/"
  cp -r ${i}/* ${SNAP_DIR_PATH}/
done
