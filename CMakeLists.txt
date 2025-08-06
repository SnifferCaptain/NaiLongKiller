cmake_minimum_required(VERSION 3.10)
project(NLKiller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖包
find_package(OpenVINO REQUIRED)
find_package(TBB REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# 设置Qt5自动处理MOC
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# 设置编译选项
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加GUI版本可执行文件
add_executable(${PROJECT_NAME}
    main.cpp
    gui.cpp
    NLKiller.cpp
)

# 链接库 - GUI版本
target_link_libraries(${PROJECT_NAME}
    openvino::runtime
    TBB::tbb
    Qt5::Core
    Qt5::Widgets
)
