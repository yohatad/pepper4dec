# dec_common-extras.cmake
#
# dec_find_onnxruntime(): find ONNX Runtime for the perception packages.
#
# By default this is ROS's onnxruntime_vendor, which ships a CPU-only build in
# /opt/ros/humble/opt/onnxruntime_vendor. For the GPU, point ONNXRUNTIME_ROOT
# (CMake variable or environment variable) at an unpacked official GPU release,
# e.g. onnxruntime-linux-x64-gpu-1.23.2, which has include/ and lib/.
#
# Keep the GPU build out of the vendor directory: apt upgrades replace that
# library with the CPU build, and a CUDA provider from one build loaded by the
# core library of another crashes (CUDNN_FE failure 11, illegal memory access).
#
# Use it in place of find_package(onnxruntime_vendor REQUIRED); keep
# onnxruntime_vendor in ament_target_dependencies() as before.

macro(dec_find_onnxruntime)
  if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()

  if(ONNXRUNTIME_ROOT)
    get_filename_component(_dec_ort_root "${ONNXRUNTIME_ROOT}" REALPATH)
    if(NOT EXISTS "${_dec_ort_root}/lib/libonnxruntime.so" OR
        NOT EXISTS "${_dec_ort_root}/include/onnxruntime_cxx_api.h")
      message(FATAL_ERROR
        "ONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT} has no lib/libonnxruntime.so "
        "and include/onnxruntime_cxx_api.h")
    endif()
    # onnxruntime_vendor only creates this target when it does not exist yet.
    if(NOT TARGET onnxruntime::onnxruntime)
      add_library(onnxruntime::onnxruntime SHARED IMPORTED)
      set_target_properties(onnxruntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${_dec_ort_root}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_dec_ort_root}/include"
        INTERFACE_LINK_OPTIONS "LINKER:-rpath,${_dec_ort_root}/lib"
      )
    endif()
  endif()

  find_package(onnxruntime_vendor REQUIRED)

  if(ONNXRUNTIME_ROOT)
    # The vendor package still exports its own headers; they must not be
    # mixed with another library version.
    set(onnxruntime_vendor_INCLUDE_DIRS "${_dec_ort_root}/include")
    set(onnxruntime_vendor_LIBRARY_DIRS "${_dec_ort_root}/lib")
    message(STATUS "ONNX Runtime: ${_dec_ort_root}")
  else()
    message(STATUS "ONNX Runtime: onnxruntime_vendor (CPU only); "
      "set ONNXRUNTIME_ROOT to a GPU release to use CUDA")
  endif()
  unset(_dec_ort_root)
endmacro()
