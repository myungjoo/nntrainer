#! /bin/bash
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2025 Donghyeon Jeong <dhyeon.jeong@samsung.com>
#
# @file prepare_opencl.sh
# @date 15 May 2025
# @brief This file is a helper tool to build android
# @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
#
# usage: ./prepare_opencl.sh target
#
# NOTE: this script downloads a prebuilt OpenCL header/stub bundle from
# nnstreamer-android-resource. Any bundled .so files MUST have ELF
# LOAD-segment alignment >= 16384 (i.e., be linked with
# -Wl,-z,max-page-size=16384) to remain loadable on 16KB-page Android
# devices (Pixel 8+, etc.) alongside nntrainer's own .so files.
# tools/package_android.sh runs an alignment audit after install and
# warns on mismatch.

set -e
TARGET=$1
TAR_PREFIX=opencl

TAR_NAME=${TAR_PREFIX}.tar.xz
URL="https://raw.githubusercontent.com/nnstreamer/nnstreamer-android-resource/main/external/${TAR_NAME}"

echo "Preparing OpenCL at ${TARGET}"

[ ! -d ${TARGET} ] && mkdir -p ${TARGET}

pushd ${TARGET}

function _download_opencl {
  [ -f $TAR_NAME ] && echo "${TAR_NAME} exists, skip downloading" && return 0
  echo "[OpenCL] downloading ${TAR_NAME}\n"
  if ! wget -q ${URL} ; then
    echo "[OpenCL] Download failed, please check url\n"
    exit $?
  fi
  echo "[OpenCL] Finish downloading OpenCL\n"
}

function _untar_opencl {
  echo "[OpenCL] untar OpenCL\n"
  tar xf ${TAR_NAME} -C ${TARGET}
  rm -f ${TAR_NAME}
}

[ ! -d "${TAR_PREFIX}" ] && _download_opencl && _untar_opencl

popd
