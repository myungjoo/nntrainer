# SPDX-License-Identifier: Apache-2.0
#
# ProGuard rules shipped with the nntrainer AAR. Applied automatically
# to consumer apps that depend on this AAR.
#
# Keep the loader entry point so reflection-based init (e.g. from
# Application.onCreate via Class.forName) keeps working.
-keep class com.samsung.nntrainer.NntrainerNative {
    public static void ensureLoaded();
    public static boolean isLoaded();
}
