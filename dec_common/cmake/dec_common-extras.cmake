# dec_common-extras.cmake
#
# dec_find_onnxruntime(): find ONNX Runtime for the perception packages,
# preferring the GPU build.
#
# 1. ONNXRUNTIME_ROOT (CMake or environment variable), if set: an unpacked
#    official ONNX Runtime release with include/ and lib/.
# 2. Otherwise <workspace>/.deps/onnxruntime-linux-x64-gpu-1.23.2, where
#    scripts/install_onnxruntime_gpu.sh puts it (also in the Docker image).
# 3. Otherwise ROS's onnxruntime_vendor, a CPU-only build, with a warning.
#
# The GPU release needs CUDA 12 and cuDNN 9 at run time. The executables get
# an RPATH to the copies in the workspace's sound venv (torch cu126 installs
# them there), so they run without LD_LIBRARY_PATH; LD_LIBRARY_PATH (e.g. a
# CUDA toolkit) is still searched when that venv is absent. Extra directories
# can be given with DEC_CUDA_LIBRARY_DIRS. When none of it works the nodes
# fall back to the CPU at run time (dec_common/onnx_session.h).
#
# Keep the GPU build out of the vendor directory: apt upgrades replace that
# library with the CPU build, and a CUDA provider from one build loaded by the
# core library of another crashes (CUDNN_FE failure 11, illegal memory access).
#
# Use it in place of find_package(onnxruntime_vendor REQUIRED); keep
# onnxruntime_vendor in ament_target_dependencies() as before.

set(DEC_ONNXRUNTIME_GPU_RELEASE "onnxruntime-linux-x64-gpu-1.23.2")

macro(dec_find_onnxruntime)
  # The workspace root: colcon installs to <ws>/install/<pkg> (isolated) or
  # <ws>/install (merged).
  set(_dec_ws "")
  get_filename_component(_dec_up1 "${CMAKE_INSTALL_PREFIX}" DIRECTORY)
  get_filename_component(_dec_up2 "${_dec_up1}" DIRECTORY)
  foreach(_dec_dir "${_dec_up2}" "${_dec_up1}")
    if(NOT _dec_ws AND EXISTS "${_dec_dir}/src")
      set(_dec_ws "${_dec_dir}")
    endif()
  endforeach()

  if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()
  if(NOT ONNXRUNTIME_ROOT AND _dec_ws AND
      EXISTS "${_dec_ws}/.deps/${DEC_ONNXRUNTIME_GPU_RELEASE}/lib/libonnxruntime.so")
    set(ONNXRUNTIME_ROOT "${_dec_ws}/.deps/${DEC_ONNXRUNTIME_GPU_RELEASE}")
  endif()

  if(ONNXRUNTIME_ROOT)
    get_filename_component(_dec_ort_root "${ONNXRUNTIME_ROOT}" REALPATH)
    if(NOT EXISTS "${_dec_ort_root}/lib/libonnxruntime.so" OR
        NOT EXISTS "${_dec_ort_root}/include/onnxruntime_cxx_api.h")
      message(FATAL_ERROR
        "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT} has no lib/libonnxruntime.so "
        "and include/onnxruntime_cxx_api.h")
    endif()

    # The CUDA provider is dlopen()ed by libonnxruntime and has no RUNPATH of
    # its own, so its CUDA/cuDNN dependencies are found through the
    # executable's DT_RPATH (not DT_RUNPATH, which only covers the
    # executable's direct dependencies): hence --disable-new-dtags.
    # The directories need not exist at build time (Docker builds the
    # workspace and the venvs in parallel stages). Call this macro before
    # creating the targets that link ONNX Runtime.
    set(_dec_cuda_dirs ${DEC_CUDA_LIBRARY_DIRS})
    if(_dec_ws)
      foreach(_dec_lib cudnn cublas cuda_runtime cufft curand)
        list(APPEND _dec_cuda_dirs
          "${_dec_ws}/.venvs/sound/lib/python3.10/site-packages/nvidia/${_dec_lib}/lib")
      endforeach()
    endif()
    # The binaries get exactly these directories, in the build tree too
    # (--symlink-install runs the build-tree binaries). The usual build RPATH
    # would add the ROS library directories ahead of LD_LIBRARY_PATH, plus an
    # empty entry (the current directory) that CMake pads it with.
    list(APPEND CMAKE_INSTALL_RPATH "${_dec_ort_root}/lib" ${_dec_cuda_dirs})
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

    # onnxruntime_vendor only creates this target when it does not exist yet.
    if(NOT TARGET onnxruntime::onnxruntime)
      add_library(onnxruntime::onnxruntime SHARED IMPORTED)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${_dec_ort_root}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_dec_ort_root}/include"
        INTERFACE_LINK_OPTIONS "LINKER:--disable-new-dtags"
      )
    endif()
  endif()

  find_package(onnxruntime_vendor REQUIRED)

  if(ONNXRUNTIME_ROOT)
    # The vendor package still exports its own headers; they must not be
    # mixed with another library version.
    set(onnxruntime_vendor_INCLUDE_DIRS "${_dec_ort_root}/include")
    set(onnxruntime_vendor_LIBRARY_DIRS "${_dec_ort_root}/lib")
    message(STATUS "ONNX Runtime: ${_dec_ort_root} (GPU)")
  else()
    message(WARNING "ONNX Runtime: onnxruntime_vendor, CPU only. For the GPU "
      "run scripts/install_onnxruntime_gpu.sh and rebuild with "
      "--cmake-force-configure (or set ONNXRUNTIME_ROOT).")
  endif()
  unset(_dec_ort_root)
  unset(_dec_ws)
  unset(_dec_up1)
  unset(_dec_up2)
  unset(_dec_dir)
  unset(_dec_lib)
  unset(_dec_cuda_dirs)
endmacro()
