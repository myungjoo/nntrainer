#!/usr/bin/env bash
# NOTE: this script downloads a prebuilt TensorFlow Lite .so set from
# nnstreamer-android-resource. The bundled .so files MUST have ELF
# LOAD-segment alignment >= 16384 (i.e., be linked with
# -Wl,-z,max-page-size=16384) to remain loadable on 16KB-page Android
# devices (Pixel 8+, etc.) alongside nntrainer's own .so files.
# tools/package_android.sh runs an alignment audit after install and
# warns on mismatch.
VERSION=$1
TARGET=$2

set -e
echo "PREPARING TENSORFLOW ${VERSION} at ${TARGET}"

if [ ! -d ${TARGET} ]; then
  mkdir -p ${TARGET}
fi

pushd ${TARGET}

#Get tensorflow
if [ ! -d "tensorflow-${VERSION}" ]; then
    if [ ! -f "tensorflow-lite-${VERSION}.tar.xz" ]; then
      echo "[TENSORFLOW-LITE] Download tensorflow-${VERSION}"
      URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/master/external/tensorflow-lite-${VERSION}.tar.xz"
      if ! wget -q ${URL} ; then
        echo "[TENSORFLOW-LITE] There was an error while downloading tflite, check if you have specified right version"
        exit $?
      fi
      echo "[TENSORFLOW-LITE] Finish downloading tensorflow-${VERSION}"
      echo "[TENSORFLOW-LITE] untar tensorflow-${VERSION}"
    fi
    mkdir -p tensorflow-${VERSION}
    tar -xf tensorflow-lite-${VERSION}.tar.xz -C tensorflow-${VERSION}
    rm "tensorflow-lite-${VERSION}.tar.xz"
else
  echo "[TENSORFLOW-LITE] folder already exist, exiting without downloading"
fi

popd
