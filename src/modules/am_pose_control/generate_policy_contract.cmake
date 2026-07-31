if(NOT DEFINED POLICY_MANIFEST OR NOT DEFINED OUTPUT_HEADER)
	message(FATAL_ERROR "POLICY_MANIFEST and OUTPUT_HEADER are required")
endif()

if(NOT EXISTS "${POLICY_MANIFEST}")
	message(FATAL_ERROR "AM Pose policy manifest '${POLICY_MANIFEST}' does not exist")
endif()

file(SIZE "${POLICY_MANIFEST}" manifest_size)

if(manifest_size GREATER 4096)
	message(FATAL_ERROR
		"AM Pose policy manifest '${POLICY_MANIFEST}' is ${manifest_size} bytes; deployment manifests must be <= 4096 bytes")
endif()

file(READ "${POLICY_MANIFEST}" manifest)

function(require_integer key output)
	string(REGEX MATCHALL "\"${key}\"[ \t\r\n]*:[ \t\r\n]*[0-9]+" matches "${manifest}")
	list(LENGTH matches match_count)

	if(NOT match_count EQUAL 1)
		message(FATAL_ERROR "AM Pose policy manifest must contain exactly one integer '${key}'")
	endif()

	string(REGEX REPLACE ".*:[ \t\r\n]*([0-9]+)$" "\\1" value "${matches}")
	set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(require_string key output)
	string(REGEX MATCHALL "\"${key}\"[ \t\r\n]*:[ \t\r\n]*\"[^\"]+\"" matches "${manifest}")
	list(LENGTH matches match_count)

	if(NOT match_count EQUAL 1)
		message(FATAL_ERROR "AM Pose policy manifest must contain exactly one string '${key}'")
	endif()

	string(REGEX REPLACE ".*:[ \t\r\n]*\"([^\"]+)\"$" "\\1" value "${matches}")
	set(${output} "${value}" PARENT_SCOPE)
endfunction()

require_integer("manifest_version" manifest_version)
require_string("policy_name" policy_name)
require_integer("observation_dim" observation_dim)
require_string("arm_observation" arm_observation)
require_integer("action_dim" action_dim)
require_string("action_semantics" action_semantics)

string(REGEX REPLACE "[ \t\r\n]" "" compact_manifest "${manifest}")

if(NOT compact_manifest MATCHES "^\\{.*\\}$"
   OR compact_manifest MATCHES "^\\{,"
   OR compact_manifest MATCHES ",\\}$"
   OR compact_manifest MATCHES ",,")
	message(FATAL_ERROR "AM Pose policy manifest must be a JSON object with comma-separated fields")
endif()

string(REGEX MATCHALL "," manifest_commas "${compact_manifest}")
list(LENGTH manifest_commas manifest_comma_count)

if(NOT manifest_comma_count EQUAL 5)
	message(FATAL_ERROR "AM Pose policy manifest must contain exactly six comma-separated fields")
endif()

set(unparsed_manifest "${compact_manifest}")
string(REGEX REPLACE "\\\"manifest_version\\\":[0-9]+" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "\\\"policy_name\\\":\\\"[^\\\"]+\\\"" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "\\\"observation_dim\\\":[0-9]+" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "\\\"arm_observation\\\":\\\"[^\\\"]+\\\"" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "\\\"action_dim\\\":[0-9]+" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "\\\"action_semantics\\\":\\\"[^\\\"]+\\\"" "" unparsed_manifest "${unparsed_manifest}")
string(REGEX REPLACE "[{},]" "" unparsed_manifest "${unparsed_manifest}")

if(NOT unparsed_manifest STREQUAL "")
	message(FATAL_ERROR "AM Pose policy manifest contains invalid JSON or unsupported fields")
endif()

if(NOT manifest_version EQUAL 1)
	message(FATAL_ERROR "AM Pose policy manifest_version must be 1, got '${manifest_version}'")
endif()

if(policy_name STREQUAL "")
	message(FATAL_ERROR "AM Pose policy manifest policy_name must not be empty")
endif()

if(NOT action_dim EQUAL 4)
	message(FATAL_ERROR "AM Pose policy manifest action_dim must be 4, got '${action_dim}'")
endif()

if(NOT action_semantics STREQUAL "normalized_throttle")
	message(FATAL_ERROR
		"AM Pose policy manifest action_semantics must be normalized_throttle, got '${action_semantics}'")
endif()

if(arm_observation STREQUAL "none")
	set(arm_observation_features 0)
	set(expected_observation_dim 130)
elseif(arm_observation STREQUAL "position")
	set(arm_observation_features 1)
	set(expected_observation_dim 134)
elseif(arm_observation STREQUAL "velocity")
	set(arm_observation_features 2)
	set(expected_observation_dim 134)
elseif(arm_observation STREQUAL "position_velocity")
	set(arm_observation_features 3)
	set(expected_observation_dim 138)
else()
	message(FATAL_ERROR
		"AM Pose policy manifest arm_observation must be none, position, velocity, or position_velocity; got '${arm_observation}'")
endif()

if(NOT observation_dim EQUAL expected_observation_dim)
	message(FATAL_ERROR
		"AM Pose policy manifest arm_observation '${arm_observation}' requires ${expected_observation_dim}D, got ${observation_dim}D")
endif()

get_filename_component(output_directory "${OUTPUT_HEADER}" DIRECTORY)
file(MAKE_DIRECTORY "${output_directory}")
file(WRITE "${OUTPUT_HEADER}" "// Generated from ${POLICY_MANIFEST}; do not edit.\n"
	"#pragma once\n"
	"#include <cstdint>\n"
	"namespace am_pose_policy_manifest {\n"
	"constexpr uint8_t kObservationDim = ${observation_dim};\n"
	"constexpr uint8_t kArmObservationFeatures = ${arm_observation_features};\n"
	"}\n")
