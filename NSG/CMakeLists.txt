cmake_minimum_required(VERSION 3.5)

project(efanna2e)
#在 CMake 中的作用是全局强制项目使用 GCC 或 Clang 编译器的旧版 C++ ABI（Application Binary Interface)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ---- LightGBM (唯一指定) ----
set(LIGHTGBM_ROOT "/home/extra_home/lxx23125236/ali/LightGBM/")

include_directories(${PROJECT_SOURCE_DIR}/include)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)

#include_directories(/home/extra_home/lxx23125236/LightGBM_install/include)
#link_directories(/home/extra_home/lxx23125236/LightGBM_install/lib)
#find_library(LIGHTGBM_LIB NAMES lib_lightgbm.so PATHS /home/extra_home/lxx23125236/LightGBM_install/lib REQUIRED)
#message(STATUS "LightGBM library: ${LIGHTGBM_LIB}")
# 4. 全局包含和链接 LightGBM (因为 DeclarativeRecall.h 在 faiss 核心中使用)
include_directories("${LIGHTGBM_ROOT}/include")


link_directories("${LIGHTGBM_ROOT}/lib")
link_libraries("${LIGHTGBM_ROOT}/lib/lib_lightgbm.so")


#set(LIGHTGBM_LIB "${LIGHTGBM_ROOT}/lib/lib_lightgbm.so")
set(LIGHTGBM_LIB /home/extra_home/lxx23125236/ali/LightGBM//lib/lib_lightgbm.so)
# Boost
find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP is required")
endif()


# Source codes
add_subdirectory(src)
add_subdirectory(tests)

# Compile flags
add_definitions(-std=c++17 -O3 -lboost -ltcmalloc_minimal -march=native -Wall -DINFO)





#cmake_minimum_required(VERSION 2.8)
#
#project(efanna2e)
#include_directories(${PROJECT_SOURCE_DIR}/include)
##OpenMP
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#else()
#    message(FATAL_ERROR "no OpenMP supprot")
#endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
#add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)
#
#add_subdirectory(src)
#add_subdirectory(tests)


#cmake_minimum_required(VERSION 2.8)
#
#project(efanna2e)
#include_directories(${PROJECT_SOURCE_DIR}/include)
##OpenMP
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#else()
#    message(FATAL_ERROR "no OpenMP supprot")
#endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free")
#add_definitions (-std=c++11 -O3 -lboost -march=native -Wall -DINFO)
#
#add_subdirectory(src)
#add_subdirectory(tests)
