project(SimpleWindowsTF_CPP)
cmake_minimum_required (VERSION 2.8.8)

set (DEPS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)
set(OpenCV_DIR ${DEPS}/opencv)
FIND_PACKAGE(OpenCV REQUIRED)

add_definitions(-DNOMINMAX -DPROTOBUF_USE_DLLS -DTF_LIB_GTL_ALIGNED_CHAR_ARRAY)


set(CPP_PRN_SRC
    main.cpp
    Utils.h
    Utils.cpp
    )
 
set(OCL_DLL_DIR ${OpenCV_DIR}/x64/${OpenCV_RUNTIME}/bin)
             
set(TF_LIB ${DEPS}/tensorflow/lib/tensorflow.lib)
set(TF_DLL_DIR ${DEPS}/tensorflow/bin/)
set(MODEL_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model/)

add_custom_target(copy_dlls ALL
                COMMAND ${CMAKE_COMMAND} -E copy "${OCL_DLL_DIR}/opencv_ffmpeg340_64.dll" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/
                COMMAND ${CMAKE_COMMAND} -E copy "${OCL_DLL_DIR}/opencv_world340.dll" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/
                COMMAND ${CMAKE_COMMAND} -E copy "${OCL_DLL_DIR}/opencv_world340d.dll" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/
                COMMAND ${CMAKE_COMMAND} -E copy "${TF_DLL_DIR}/tensorflow.dll" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/model/
                COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_FILES_DIR}/model.pb" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/model/
                COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_FILES_DIR}/labels_map.pbtxt" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/model/
                COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Ex.mp4" ${CMAKE_BINARY_DIR}/$<CONFIGURATION>/
                )

add_executable(SimpleWindowsTF_CPP ${CPP_PRN_SRC})

include_directories(${DEPS}/tensorflow/include)

add_dependencies(SimpleWindowsTF_CPP copy_dlls)


TARGET_LINK_LIBRARIES(SimpleWindowsTF_CPP ${OpenCV_LIBS} ${TF_LIB})