# GGML Installation Fix for NNTrainer

## Problem
When building nntrainer with GGML enabled using:
```bash
meson build -Denable-ggml=true && ninja -C build
```

The GGML libraries were being built successfully via the CMake subproject, but they were not being installed when running:
```bash
ninja -C build install
```

## Root Cause
The issue was that Meson's CMake subproject support builds the CMake project correctly, but it doesn't automatically install the CMake project's artifacts when `ninja install` is run. The install scripts needed to be explicitly configured in the `meson.build` file.

## Solution
Added install scripts to the GGML configuration section in `meson.build` that:

1. **Install GGML shared libraries**: Finds and copies all `*.so*` files from the CMake build directory to the nntrainer library directory
2. **Install GGML headers**: Copies headers from the GGML include directory
3. **Install additional headers**: Finds and copies any `.h` files from the GGML src directory

## Changes Made
Modified the `meson.build` file around lines 580-599 to add the following install scripts for non-Android platforms:

```meson
else
  # Add install scripts for GGML libraries and headers for non-Android platforms
  ggml_build_libdir = meson.build_root() / 'subprojects' / 'ggml' / 'src'
  ggml_source_includedir = ggml_root / 'include'
  
  # Install GGML shared libraries
  meson.add_install_script(
    'sh', '-c', 'find @0@ -name "*.so*" -exec cp {} ${DESTDIR}@1@ \\;'.format(ggml_build_libdir, nntrainer_libdir)
  )
  
  # Install GGML headers
  meson.add_install_script(
    'sh', '-c', 'cp -r @0@/* ${DESTDIR}@1@/'.format(ggml_source_includedir, nntrainer_includedir)
  )
  
  # Install additional GGML headers from src directory  
  ggml_source_srcdir = ggml_root / 'src'
  meson.add_install_script(
    'sh', '-c', 'find @0@ -name "*.h" -exec cp {} ${DESTDIR}@1@/ \\;'.format(ggml_source_srcdir, nntrainer_includedir)
  )
endif
```

## Testing
After applying this fix, you should be able to:

1. Build with GGML: `meson build -Denable-ggml=true && ninja -C build`
2. Install successfully: `ninja -C build install`
3. Verify that GGML libraries and headers are properly installed in the target directories

## Notes
- This fix applies to non-Android and non-Windows platforms (Linux, etc.)
- The Android platform already had special handling for GGML installation
- The Windows platform uses pre-built GGML libraries, so no additional install scripts were needed
- The install scripts use standard shell commands that should work on most Unix-like systems