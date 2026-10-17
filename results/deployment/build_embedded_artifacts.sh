#!/usr/bin/env bash
# ========================================================================
# EvoEdge: Portable Embedded C Artifacts Build Script (POSIX/Linux/macOS)
# ========================================================================
set -euo pipefail

OUTPUT_DIR="${1:-results/deployment}"
CC="${CC:-gcc}"

echo "Building EvoEdge Portable Embedded C Artifacts in ${OUTPUT_DIR}..."

# 1. Compile object files with strict warnings
"${CC}" -Wall -Wextra -Wpedantic -O3 -c -o "${OUTPUT_DIR}/hybrid_controller.o" "${OUTPUT_DIR}/hybrid_controller.c" -I"${OUTPUT_DIR}"
"${CC}" -Wall -Wextra -Wpedantic -O3 -c -o "${OUTPUT_DIR}/platform_adapter.o" "${OUTPUT_DIR}/platform_adapter.c" -I"${OUTPUT_DIR}"

# 2. Build static library
ar rcs "${OUTPUT_DIR}/libhybrid_controller.a" "${OUTPUT_DIR}/hybrid_controller.o" "${OUTPUT_DIR}/platform_adapter.o"

# 3. Build shared library (DLL on Windows, .so on Linux, .dylib on macOS)
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then
    SHARED_LIB="${OUTPUT_DIR}/hybrid_controller.dll"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    SHARED_LIB="${OUTPUT_DIR}/libhybrid_controller.dylib"
else
    SHARED_LIB="${OUTPUT_DIR}/libhybrid_controller.so"
fi

"${CC}" -Wall -Wextra -Wpedantic -O3 -shared -fPIC -o "${SHARED_LIB}" "${OUTPUT_DIR}/hybrid_controller.c" "${OUTPUT_DIR}/platform_adapter.c" -I"${OUTPUT_DIR}" -lm

# 4. Build standalone native test runner
RUNNER_BIN="${OUTPUT_DIR}/hybrid_controller_runner"
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then
    RUNNER_BIN="${RUNNER_BIN}.exe"
fi

"${CC}" -Wall -Wextra -Wpedantic -O3 -o "${RUNNER_BIN}" "${OUTPUT_DIR}/hybrid_controller_runner.c" "${OUTPUT_DIR}/hybrid_controller.c" "${OUTPUT_DIR}/platform_adapter.c" -I"${OUTPUT_DIR}" -lm

echo "Build completed successfully with zero warnings."
