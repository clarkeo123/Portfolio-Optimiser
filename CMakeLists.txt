cmake_minimum_required(VERSION 3.23)

project(PortfolioOptimizer VERSION 1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_executable(portfolio_optimizer
    src/main.cpp
    src/MarketData.cpp
    src/Portfolio.cpp
    src/GeneticAlgorithm.cpp
)

target_include_directories(portfolio_optimizer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(portfolio_optimizer
    PRIVATE
        Eigen3::Eigen
)

target_compile_options(portfolio_optimizer
    PRIVATE
        $<$<CONFIG:Release>:-O3 -DNDEBUG>
)