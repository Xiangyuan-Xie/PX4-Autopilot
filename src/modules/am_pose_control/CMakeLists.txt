############################################################################
#
#   Copyright (c) 2026.
#
############################################################################

add_compile_options(-Wno-float-equal)
get_directory_property(FLAGS COMPILE_OPTIONS)
list(REMOVE_ITEM FLAGS "-Wcast-align")
set_directory_properties(PROPERTIES COMPILE_OPTIONS "${FLAGS}")

px4_add_module(
	MODULE am_pose_control
	MAIN am_pose_control
	COMPILE_FLAGS
	SRCS
		am_pose_control.cpp
		am_pose_control.hpp
		am_pose_trajectory.cpp
		am_pose_trajectory.hpp
		am_pose_policy_adapter.cpp
		am_pose_policy_adapter.hpp
		../mc_pos_control/Takeoff/Takeoff.cpp
	DEPENDS
		px4_work_queue
		hysteresis
		mathlib
		motion_planning
)

file(GLOB AM_POSE_POLICY_BLOB CONFIGURE_DEPENDS
	${CMAKE_CURRENT_SOURCE_DIR}/blob/policy.h
)

if(AM_POSE_POLICY_BLOB)
	set(AM_POSE_POLICY_MANIFEST ${CMAKE_CURRENT_SOURCE_DIR}/blob/policy.json)
	set(AM_POSE_POLICY_CONTRACT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/am_pose_policy_contract.hpp)

	if(NOT EXISTS ${AM_POSE_POLICY_MANIFEST})
		message(FATAL_ERROR "AM Pose policy.h requires blob/policy.json deployment manifest")
	endif()

	add_custom_command(
		OUTPUT ${AM_POSE_POLICY_CONTRACT_HEADER}
		COMMAND ${CMAKE_COMMAND}
			-DPOLICY_MANIFEST=${AM_POSE_POLICY_MANIFEST}
			-DOUTPUT_HEADER=${AM_POSE_POLICY_CONTRACT_HEADER}
			-P ${CMAKE_CURRENT_SOURCE_DIR}/generate_policy_contract.cmake
		DEPENDS ${AM_POSE_POLICY_MANIFEST} ${CMAKE_CURRENT_SOURCE_DIR}/generate_policy_contract.cmake
		COMMENT "Generating AM Pose policy observation contract"
		VERBATIM
	)
	add_custom_target(am_pose_control_policy_contract DEPENDS ${AM_POSE_POLICY_CONTRACT_HEADER})
	add_dependencies(am_pose_control am_pose_control_policy_contract)
	target_sources(am_pose_control PRIVATE ${AM_POSE_POLICY_BLOB} ${AM_POSE_POLICY_CONTRACT_HEADER})
	set_source_files_properties(am_pose_policy_adapter.cpp am_pose_control_policy_blob_test.cpp
		PROPERTIES OBJECT_DEPENDS "${AM_POSE_POLICY_BLOB};${AM_POSE_POLICY_CONTRACT_HEADER}")
endif()

target_include_directories(am_pose_control PRIVATE
	${PX4_SOURCE_DIR}/src/modules/mc_pos_control/Takeoff
	${PX4_SOURCE_DIR}/third_party/rl_tools/include
	${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_compile_features(am_pose_control PRIVATE cxx_std_17)
target_compile_options(am_pose_control PRIVATE -Wno-unused-local-typedefs)

add_library(am_pose_control_policy_blob INTERFACE)
target_include_directories(am_pose_control_policy_blob INTERFACE
	${CMAKE_CURRENT_SOURCE_DIR}
	${PX4_SOURCE_DIR}/third_party/rl_tools/include
	${CMAKE_CURRENT_BINARY_DIR}/generated
)

if(TARGET am_pose_control_policy_contract)
	add_dependencies(am_pose_control_policy_blob am_pose_control_policy_contract)
endif()
target_compile_features(am_pose_control_policy_blob INTERFACE cxx_std_17)
target_compile_options(am_pose_control_policy_blob INTERFACE -Wno-unused-local-typedefs)

px4_add_unit_gtest(
	SRC am_pose_control_policy_blob_test.cpp
	EXTRA_SRCS am_pose_policy_adapter.cpp
	LINKLIBS am_pose_control_policy_blob
)
px4_add_unit_gtest(
	SRC am_pose_control_test.cpp
	EXTRA_SRCS ../mc_pos_control/Takeoff/Takeoff.cpp am_pose_trajectory.cpp
	INCLUDES
		${PX4_SOURCE_DIR}/src/modules/mc_pos_control/Takeoff
	LINKLIBS am_pose_control_policy_blob hysteresis motion_planning
)
px4_add_unit_gtest(
	SRC am_pose_trajectory_test.cpp
	EXTRA_SRCS am_pose_trajectory.cpp
	LINKLIBS motion_planning
)

if(BUILD_TESTING)
	add_test(NAME am_pose_policy_manifest_test
		COMMAND ${CMAKE_COMMAND}
			-DGENERATOR_SCRIPT=${CMAKE_CURRENT_SOURCE_DIR}/generate_policy_contract.cmake
			-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/am_pose_policy_manifest_test
			-P ${CMAKE_CURRENT_SOURCE_DIR}/generate_policy_contract_test.cmake)
endif()
