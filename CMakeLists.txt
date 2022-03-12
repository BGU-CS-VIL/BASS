cmake_minimum_required(VERSION 2.8)
project( Superpixels )
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED )
set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Xptxas -O3")

SET(OpenCV_DIR "/installation/OpenCV-3.4.4/share/OpenCV")

INCLUDE_DIRECTORIES(
	${OpenCV_DIR} 
	include
)

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

cuda_add_executable(
	Sp_demo_for_direc
	demo_for_direc.cpp
	Superpixels.cpp 
	my_sp_struct.h
	RgbLab.h
	RgbLab.cu
	init_seg.h
	init_seg.cu
	sp_helper.h
	sp_helper.cu
	update_param.h
	update_param.cu
	update_seg.h
	update_seg.cu
	utils.cpp
	utils.h
	s_m.h
	s_m.cu
)

TARGET_LINK_LIBRARIES(Sp_demo_for_direc ${OpenCV_LIBS})

find_package(Threads)

