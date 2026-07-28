cmake_minimum_required(VERSION 3.23)

project(PortfolioOptimizer VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen.
find_package(Eigen3 REQUIRED)

# Create the executable.
add_executable(portfolio_optimizer
    src/main.cpp
    src/MarketData.cpp
    src/Portfolio.cpp
)

# Tell the compiler where our headers are.
target_include_directories(portfolio_optimizer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link Eigen.
target_link_libraries(portfolio_optimizer
    PRIVATE
        Eigen3::Eigen
)