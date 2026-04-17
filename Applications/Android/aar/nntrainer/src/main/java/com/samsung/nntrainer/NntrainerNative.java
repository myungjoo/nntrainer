// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2026 nntrainer contributors
 *
 * @file   NntrainerNative.java
 * @brief  Single entry point for loading nntrainer's native libraries.
 *
 * Consumer apps should call NntrainerNative.ensureLoaded() once,
 * preferably from Application.onCreate() or a static initializer in
 * the class that first uses the C/C++ API. This guarantees that the
 * .so files bundled inside the AAR are loaded in the right
 * dependency order before any JNI method dispatches.
 *
 * Why this class exists at all:
 *   * libnntrainer.so DT_NEEDED's libomp.so (since the
 *     "[Android][high] drop -static-openmp" change). On older
 *     bionic/linker combinations, lazy DT_NEEDED resolution can
 *     race with parallel System.loadLibrary calls; loading deps
 *     first in an explicit order avoids that.
 *   * The consumer would otherwise have to know the dependency
 *     graph (libomp <- libnntrainer <- libccapi-nntrainer <-
 *     libcapi-nntrainer) which is internal to the AAR.
 */
package com.samsung.nntrainer;

public final class NntrainerNative {

    private static volatile boolean loaded = false;

    private NntrainerNative() { /* no instances */ }

    /**
     * Load nntrainer's native libraries in dependency order. Idempotent
     * and thread-safe. Throws UnsatisfiedLinkError if a required .so is
     * missing from the APK's lib/&lt;abi&gt;/ directory.
     */
    public static synchronized void ensureLoaded() {
        if (loaded) {
            return;
        }
        // Order matters: bottom of the dep graph first.
        System.loadLibrary("omp");
        System.loadLibrary("openblas");
        System.loadLibrary("nntrainer");
        System.loadLibrary("ccapi-nntrainer");
        System.loadLibrary("capi-nntrainer");
        loaded = true;
    }

    /** True once {@link #ensureLoaded()} has completed successfully. */
    public static boolean isLoaded() {
        return loaded;
    }
}
