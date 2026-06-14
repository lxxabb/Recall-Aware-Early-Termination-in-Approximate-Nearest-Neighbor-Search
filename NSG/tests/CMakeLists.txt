set(CMAKE_CXX_STANDARD 17)

# 1. 关键：让可执行文件运行时优先找我们自己的 LightGBM 库
#set(CMAKE_INSTALL_RPATH "/home/extra_home/lxx23125236/LightGBM_install/lib")
#set(CMAKE_BUILD_RPATH   "/home/extra_home/lxx23125236/LightGBM_install/lib")
set(LIGHTGBM_ROOT "/home/extra_home/lxx23125236/ali/LightGBM/")
# 1. 关键：让可执行文件运行时优先找我们自己的 LightGBM 库
set(CMAKE_INSTALL_RPATH "${LIGHTGBM_ROOT}/lib")
set(CMAKE_BUILD_RPATH   "${LIGHTGBM_ROOT}/lib")

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/opt/gperftools")


add_executable(test_nsg_index test_nsg_index.cpp)
find_library(TCMALLOC_LIB tcmalloc_minimal REQUIRED)
target_link_libraries(test_nsg_index ${PROJECT_NAME} ${TCMALLOC_LIB} ${LIGHTGBM_LIB})
#target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)

add_executable(test_nsg_search test_nsg_search.cpp)
target_link_libraries(test_nsg_search ${PROJECT_NAME} ${LIGHTGBM_LIB})

add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} ${TCMALLOC_LIB} ${LIGHTGBM_LIB})


#set(CMAKE_CXX_STANDARD 11)
#
#add_executable(test_nsg_index test_nsg_index.cpp)
#target_link_libraries(test_nsg_index ${PROJECT_NAME} -ltcmalloc)
#
#add_executable(test_nsg_search test_nsg_search.cpp)
#target_link_libraries(test_nsg_search ${PROJECT_NAME})
#
#add_executable(test_nsg_optimized_search test_nsg_optimized_search.cpp)
#target_link_libraries(test_nsg_optimized_search ${PROJECT_NAME} -ltcmalloc)

