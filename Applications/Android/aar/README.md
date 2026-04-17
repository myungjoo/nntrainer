# nntrainer Android AAR (skeleton)

This directory is a `com.android.library` gradle project that produces a
distributable `nntrainer-release.aar`. It is **scaffolding** — enough to
build an AAR locally and consume it in a separate Android project, but
not yet wired into CI, Maven publication, or version pinning.

## What it is

A single library module (`:nntrainer`) that bundles:

- the nntrainer native library set built by `tools/package_android.sh`
  (`libnntrainer.so`, `libccapi-nntrainer.so`, `libcapi-nntrainer.so`,
  `libopenblas.so`, `libomp.so`, optionally `libOpenCL.so`);
- a single Java entry point, `com.samsung.nntrainer.NntrainerNative`,
  whose `ensureLoaded()` method calls `System.loadLibrary` in the right
  dependency order;
- consumer ProGuard rules that keep `NntrainerNative` reachable.

Layout:

```
Applications/Android/aar/
├── README.md                   (this file)
├── build.gradle                (top-level)
├── settings.gradle
├── gradle.properties
└── nntrainer/                  (the library module)
    ├── build.gradle
    ├── consumer-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── java/com/samsung/nntrainer/NntrainerNative.java
        └── jniLibs/arm64-v8a/  (populated at build time, .gitkeep only)
```

## How to build (local)

```sh
# 1. From the repo root, build the native libs.
./tools/package_android.sh

# 2. From this directory, assemble the AAR.
cd Applications/Android/aar
./gradlew :nntrainer:assembleRelease
```

The AAR lands in `nntrainer/build/outputs/aar/nntrainer-release.aar`.

## How to consume (local)

In the consumer app's `build.gradle`:

```gradle
dependencies {
    implementation files('libs/nntrainer-release.aar')
}
```

In code:

```java
NntrainerNative.ensureLoaded();           // once, before any JNI dispatch
// ... call into ccapi/capi from your own JNI bridge ...
```

## What is NOT yet done (intentionally — TODO list)

- [ ] **CI integration.** `.github/workflows/android.yml` does not yet
      build this module. Adding a `gradle :nntrainer:assembleRelease`
      step would produce an AAR artifact per PR.
- [ ] **Maven publication.** No `maven-publish` plugin, no
      `groupId`/`artifactId`/`version` coordinates declared. Decide:
      Sonatype OSSRH? GitHub Packages? A vendored maven repo?
- [ ] **Version pinning.** `versionName` is implied; no explicit
      `versionCode`/`versionName` block in
      `nntrainer/build.gradle`. Should be wired to nntrainer's release
      versioning.
- [ ] **Multiple ABIs.** Currently arm64-v8a only, matching
      `tools/package_android.sh` and `jni/Application.mk`. armeabi-v7a
      / x86_64 would each require a separate native build pass and
      `abiFilters` adjustment.
- [ ] **Symbol visibility / API contract.** The AAR currently exposes
      one Java class (`NntrainerNative`) but does not provide a
      Java-level API around the native CCAPI/CAPI; consumers must
      still write their own JNI bridge. A higher-level Java/Kotlin
      facade over `nntrainer/api/ccapi` would make the AAR
      self-sufficient.
- [ ] **Sample consumer.** No project yet demonstrates consuming the
      AAR (the existing `Applications/Android/{ResnetJNI,
      PicoGPTJNI, NNDetector, kotlin}` apps consume the .so files
      directly from `libs/arm64-v8a/`, not via the AAR).
This commit is deliberately scope-limited: it gets the directory
layout, manifest stub, module gradle config, gradle wrapper, Java
loader entry point, and CI hookup in place so the rest of the items
above can land as separate PRs that each have a clear, single-concern
review.

CI **does** exercise this module: `.github/workflows/android.yml`
runs `./gradlew :nntrainer:assembleRelease` after the native build
in both the `-Denable-opencl=false` and `-Denable-opencl=true` matrix
arms, and uploads the produced AAR as a workflow artifact.
