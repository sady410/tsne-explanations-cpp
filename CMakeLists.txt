cmake_minimum_required(VERSION 3.16)

project(tsne_explanations_cpp VERSION 0.1.0 LANGUAGES CXX)

option(TSNE_ENABLE_OPENMP "Use OpenMP parallelization when available" ON)

add_executable(tsne src/tsne.cpp)
add_executable(tsne_explainer src/tsne_explainer.cpp)

foreach(target IN ITEMS tsne tsne_explainer)
    target_compile_features(${target} PRIVATE cxx_std_17)

    if(MSVC)
        target_compile_options(${target} PRIVATE /O2 /W4)
    else()
        target_compile_options(${target} PRIVATE -O3 -Wall -Wextra -Wpedantic)
    endif()
endforeach()

if(TSNE_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tsne PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(tsne_explainer PRIVATE OpenMP::OpenMP_CXX)
        # Some MinGW/CMake combinations detect -fopenmp but leave the imported
        # target's link interface empty. Passing the driver flag fixes that
        # case and is also valid for GNU on other platforms.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_options(tsne PRIVATE -fopenmp)
            target_link_options(tsne_explainer PRIVATE -fopenmp)
        endif()
        message(STATUS "OpenMP enabled")
    else()
        message(STATUS "OpenMP not found; building a single-threaded version")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS tsne tsne_explainer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
