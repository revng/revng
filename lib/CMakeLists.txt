#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

#
# Handle the analysis libraries
#

add_custom_target(analyses)

macro(revng_add_analyses_library_internal NAME)
  revng_add_analyses_library("${NAME}" revng ${ARGN})
  add_dependencies(analyses "${NAME}")
endmacro()

macro(revng_add_library_internal NAME TYPE)
  revng_add_library("${NAME}" "${TYPE}" revng ${ARGN})
endmacro()

add_subdirectory(ABI)
add_subdirectory(Backend)
add_subdirectory(BasicAnalyses)
add_subdirectory(Canonicalize)
add_subdirectory(DataLayoutAnalysis)
add_subdirectory(EarlyFunctionAnalysis)
add_subdirectory(FunctionCallIdentification)
add_subdirectory(FunctionIsolation)
add_subdirectory(GraphLayout)
add_subdirectory(HeadersGeneration)
add_subdirectory(ImportFromC)
add_subdirectory(InitModelTypes)
add_subdirectory(LocalVariables)
add_subdirectory(LLMRename)
add_subdirectory(Lift)
add_subdirectory(mlir)
add_subdirectory(Model)
add_subdirectory(Pipebox)
add_subdirectory(PipeboxCommon)
add_subdirectory(Pipeline)
add_subdirectory(PipelineC)
add_subdirectory(Pipes)
add_subdirectory(PromoteStackPointer)
add_subdirectory(PTML)
add_subdirectory(Recompile)
add_subdirectory(RemoveExtractValues)
add_subdirectory(RemoveLiftingArtifacts)
add_subdirectory(RestructureCFG)
add_subdirectory(Storage)
add_subdirectory(Support)
add_subdirectory(TypeNames)
add_subdirectory(TypeShrinking)
add_subdirectory(UnitTestHelpers)
add_subdirectory(ValueMaterializer)
add_subdirectory(Yield)
