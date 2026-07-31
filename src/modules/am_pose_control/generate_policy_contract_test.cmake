if(NOT DEFINED GENERATOR_SCRIPT OR NOT DEFINED TEST_ROOT)
	message(FATAL_ERROR "GENERATOR_SCRIPT and TEST_ROOT are required")
endif()

file(MAKE_DIRECTORY "${TEST_ROOT}")

function(run_valid_manifest name observation_dim arm_observation expected_features)
	set(manifest "${TEST_ROOT}/${name}.json")
	set(header "${TEST_ROOT}/${name}.hpp")
	file(WRITE "${manifest}" "{\n"
		"  \"manifest_version\": 1,\n"
		"  \"policy_name\": \"${name}\",\n"
		"  \"observation_dim\": ${observation_dim},\n"
		"  \"arm_observation\": \"${arm_observation}\",\n"
		"  \"action_dim\": 4,\n"
		"  \"action_semantics\": \"normalized_throttle\"\n"
		"}\n")
	execute_process(
		COMMAND "${CMAKE_COMMAND}" -DPOLICY_MANIFEST=${manifest} -DOUTPUT_HEADER=${header} -P ${GENERATOR_SCRIPT}
		RESULT_VARIABLE result
		OUTPUT_VARIABLE output
		ERROR_VARIABLE error)

	if(NOT result EQUAL 0)
		message(FATAL_ERROR "valid ${name} manifest failed: ${output}${error}")
	endif()

	file(READ "${header}" generated)
	string(FIND "${generated}" "kObservationDim = ${observation_dim}" dimension_index)
	string(FIND "${generated}" "kArmObservationFeatures = ${expected_features}" feature_index)

	if(dimension_index EQUAL -1 OR feature_index EQUAL -1)
		message(FATAL_ERROR "${name} contract header did not contain the expected values")
	endif()
endfunction()

function(expect_manifest_failure name body expected_error)
	set(manifest "${TEST_ROOT}/${name}.json")
	set(header "${TEST_ROOT}/${name}.hpp")
	file(WRITE "${manifest}" "${body}")
	execute_process(
		COMMAND "${CMAKE_COMMAND}" -DPOLICY_MANIFEST=${manifest} -DOUTPUT_HEADER=${header} -P ${GENERATOR_SCRIPT}
		RESULT_VARIABLE result
		OUTPUT_VARIABLE output
		ERROR_VARIABLE error)

	if(result EQUAL 0)
		message(FATAL_ERROR "invalid ${name} manifest unexpectedly succeeded")
	endif()

	string(FIND "${output}${error}" "${expected_error}" error_index)

	if(error_index EQUAL -1)
		message(FATAL_ERROR "invalid ${name} manifest did not report '${expected_error}': ${output}${error}")
	endif()
endfunction()

run_valid_manifest(no_arm 130 none 0)
run_valid_manifest(position 134 position 1)
run_valid_manifest(velocity 134 velocity 2)
run_valid_manifest(position_velocity 138 position_velocity 3)

expect_manifest_failure(missing_arm_observation
"{\n  \"manifest_version\": 1,\n  \"policy_name\": \"missing\",\n  \"observation_dim\": 134,\n  \"action_dim\": 4,\n  \"action_semantics\": \"normalized_throttle\"\n}\n"
"arm_observation")
expect_manifest_failure(mismatched_dimension
"{\n  \"manifest_version\": 1,\n  \"policy_name\": \"mismatch\",\n  \"observation_dim\": 130,\n  \"arm_observation\": \"position\",\n  \"action_dim\": 4,\n  \"action_semantics\": \"normalized_throttle\"\n}\n"
"requires 134D")
expect_manifest_failure(wrong_action_dim
"{\n  \"manifest_version\": 1,\n  \"policy_name\": \"action\",\n  \"observation_dim\": 130,\n  \"arm_observation\": \"none\",\n  \"action_dim\": 3,\n  \"action_semantics\": \"normalized_throttle\"\n}\n"
"action_dim must be 4")
expect_manifest_failure(unknown_arm_observation
"{\n  \"manifest_version\": 1,\n  \"policy_name\": \"unknown-arm\",\n  \"observation_dim\": 134,\n  \"arm_observation\": \"effort\",\n  \"action_dim\": 4,\n  \"action_semantics\": \"normalized_throttle\"\n}\n"
"got 'effort'")
expect_manifest_failure(malformed_json
"{\n  \"manifest_version\": 1\n  \"policy_name\": \"malformed\",\n  \"observation_dim\": 130,\n  \"arm_observation\": \"none\",\n  \"action_dim\": 4,\n  \"action_semantics\": \"normalized_throttle\"\n}\n"
"exactly six comma-separated fields")

execute_process(
	COMMAND "${CMAKE_COMMAND}" -DPOLICY_MANIFEST=${TEST_ROOT}/missing.json -DOUTPUT_HEADER=${TEST_ROOT}/missing.hpp -P ${GENERATOR_SCRIPT}
	RESULT_VARIABLE missing_result
	OUTPUT_VARIABLE missing_output
	ERROR_VARIABLE missing_error)
if(missing_result EQUAL 0)
	message(FATAL_ERROR "missing manifest unexpectedly succeeded")
endif()
string(FIND "${missing_output}${missing_error}" "does not exist" missing_error_index)
if(missing_error_index EQUAL -1)
	message(FATAL_ERROR "missing manifest did not report that the file is absent")
endif()

set(oversized_manifest "${TEST_ROOT}/oversized.json")
file(WRITE "${oversized_manifest}" "{")
foreach(index RANGE 1 4097)
	file(APPEND "${oversized_manifest}" "x")
endforeach()
file(APPEND "${oversized_manifest}" "}")
execute_process(
	COMMAND "${CMAKE_COMMAND}" -DPOLICY_MANIFEST=${oversized_manifest} -DOUTPUT_HEADER=${TEST_ROOT}/oversized.hpp -P ${GENERATOR_SCRIPT}
	RESULT_VARIABLE oversized_result
	OUTPUT_VARIABLE oversized_output
	ERROR_VARIABLE oversized_error)
if(oversized_result EQUAL 0)
	message(FATAL_ERROR "oversized manifest unexpectedly succeeded")
endif()
string(FIND "${oversized_output}${oversized_error}" "must be <= 4096 bytes" oversized_error_index)
if(oversized_error_index EQUAL -1)
	message(FATAL_ERROR "oversized manifest did not report the size limit")
endif()
