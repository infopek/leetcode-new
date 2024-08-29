cmake_minimum_required(VERSION 3.16..3.24)

project(LeetCode 	VERSION 0.0.1
                        DESCRIPTION "LeetCode project"
                        LANGUAGES CXX)     

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(LeetCode
    src/models/list_node.h
    src/models/tree_node.h

    src/main.cpp
)


