APP_ABI           := arm64-v8a
LIBCXX_USE_GABIXX := true
APP_STL           := c++_shared
# Bump from android-29 to android-33: bionic/linker changes between
# API 29 and API 31+ tightened RELRO, FORTIFY, and dlopen namespace
# checks; binaries stamped at the older API level skip those guarantees
# and trip the newer validators on Pixel/Galaxy devices running
# Android 12+. android-33 also matches the `targetSdk 31`/`compileSdk 31+`
# used by the sample apps, so the binary contract matches the Java side.
APP_PLATFORM      := android-33
APP_SUPPORT_FLEXIBLE_PAGE_SIZES := true
