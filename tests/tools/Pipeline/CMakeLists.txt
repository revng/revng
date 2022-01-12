#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(StringContainerLibrary SHARED ${CMAKE_CURRENT_LIST_DIR}/StringContainerLibrary.cpp)
target_link_libraries(StringContainerLibrary revngPipeline revngPipes)

macro(add_auto_Pipe_test TestName PipelineFile Targets InputsList OutputList Flags)
	add_test(NAME ${TestName} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/revng-pipeline -P ${CMAKE_CURRENT_LIST_DIR}/${PipelineFile} ${Targets} -i ${InputsList} -o ${OutputList} -l $<TARGET_FILE:StringContainerLibrary> -p ${CMAKE_CURRENT_BINARY_DIR}/${TestName} -f ${Flags})
endmacro()

macro(add_auto_Pipe_dump_test TestName PipelineFile) 
	add_test(NAME ${TestName} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/revng-pipeline -P ${CMAKE_CURRENT_LIST_DIR}/${PipelineFile} dc:dc:Root -l $<TARGET_FILE:StringContainerLibrary> -d)
endmacro()

macro(add_auto_Pipe_test_with_output TestName PipelineFile Targets InputsList OutputList Flags ToCheckFile ExpectedOutput)

	add_auto_Pipe_test(${TestName} ${PipelineFile} ${Targets} ${InputsList} ${OutputList} ${Flags})
	add_test(NAME check-${TestName} COMMAND diff ${ToCheckFile} ${ExpectedOutput})

 set_tests_properties(check-${TestName} PROPERTIES DEPENDS ${TestName})
endmacro()

macro(add_auto_Pipe_failing_test TestName PipelineFile Targets InputsList OutputList Flags)

	add_auto_Pipe_test(${TestName} ${PipelineFile} ${Targets} ${InputsList} ${OutputList} ${Flags} ${ExpectedOutput})
    set_tests_properties(${TestName} PROPERTIES WILL_FAIL TRUE)
endmacro()


set(CopyPipeTestInputs "FirstStep:Strings1:${CMAKE_CURRENT_LIST_DIR}/CopyPipeTestInput.txt")
set(CopyPipeTestOutputs "End:Strings2:${CMAKE_CURRENT_BINARY_DIR}/CopyPipeTestOutput.txt")
add_auto_Pipe_test_with_output(copy-Pipe-run-test CopyPipeTestPipeline.yml Strings2:Root:Root ${CopyPipeTestInputs} ${CopyPipeTestOutputs} None CopyPipeTestOutput.txt ${CMAKE_CURRENT_LIST_DIR}/CopyPipeTestExpected.txt)

add_auto_Pipe_test_with_output(flag-pipeline-test CopyPipeTestPipeline.yml Strings2:Root:Root ${CopyPipeTestInputs} ${CopyPipeTestOutputs} None CopyPipeTestOutput.txt ${CMAKE_CURRENT_LIST_DIR}/CopyPipeTestExpected.txt)


add_auto_Pipe_failing_test(flag-auto-Pipe-fail-test CopyPipeFlagTestPipeline.yml Strings2:Root:Root ${CopyPipeTestInputs} ${CopyPipeTestOutputs} EnablingFlags)

add_auto_Pipe_failing_test(missing-pass-pipeline-fail-test MissingPassPipelineTest.yml Strings2:Root:Root ${CopyPipeTestInputs} ${CopyPipeTestOutputs} DontCare)

add_auto_Pipe_dump_test(llvm-pass-test PassPipelineTest.yml) 

add_test(NAME MultiStepPipelineTest 
	     COMMAND 
		   ${CMAKE_CURRENT_LIST_DIR}/MultiStepPipelineTest.sh
		   ${CMAKE_CURRENT_LIST_DIR}/MultiStepPipeline.yml
		   ${CMAKE_CURRENT_LIST_DIR}/MultiStepPipelineInput.txt
		   ${CMAKE_CURRENT_BINARY_DIR}/MultiStepPipelineOut.txt
		   ${CMAKE_CURRENT_BINARY_DIR}/MultiStepPipelineTestDir/
		   -l $<TARGET_FILE:StringContainerLibrary> 
		   ${CMAKE_CURRENT_LIST_DIR}/MultiStepPipelineOutput.txt
		   )
set_tests_properties(MultiStepPipelineTest PROPERTIES DEPENDS revng-pipeline)

revng_add_library_internal(FunctionCounter SHARED ${CMAKE_CURRENT_LIST_DIR}/ModelFunctionCounter.cpp)
target_link_libraries(FunctionCounter revngPipeline)
