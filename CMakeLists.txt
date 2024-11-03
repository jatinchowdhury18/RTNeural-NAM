cmake_minimum_required(VERSION 3.29)
project(RTNeural_NAM)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(RTNeural)
add_subdirectory(NeuralAmpModelerCore)

add_executable(RTNeural_NAM)
target_sources(RTNeural_NAM
    PRIVATE
        main.cpp
        NeuralAmpModelerCore/NAM/activations.cpp
        NeuralAmpModelerCore/NAM/convnet.cpp
        NeuralAmpModelerCore/NAM/dsp.cpp
        NeuralAmpModelerCore/NAM/get_dsp.cpp
        NeuralAmpModelerCore/NAM/lstm.cpp
        NeuralAmpModelerCore/NAM/util.cpp
        NeuralAmpModelerCore/NAM/wavenet.cpp
)
target_compile_definitions(RTNeural_NAM
    PRIVATE
        ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_compile_definitions(NeuralAmpModelerCore PUBLIC NAM_SAMPLE_FLOAT=1)

target_include_directories(RTNeural_NAM
    PRIVATE
        NeuralAmpModelerCore
)
target_link_libraries(RTNeural_NAM
    PRIVATE
        RTNeural
)
