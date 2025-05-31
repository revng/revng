revng_add_analyses_library_internal(
  revngMLIRPipes
  CBackendPipe.cpp
  ImportCliftTypesPipe.cpp
  LLVMToCliftPipe.cpp
  MLIRContainer.cpp)

target_link_libraries(
  revngMLIRPipes
  PUBLIC revngPipeline
         revngPipes
         MLIRTransforms
         MLIRDialect
         MLIRIR
         MLIRBytecodeReader
         MLIRBytecodeWriter
         MLIRParser
         MLIRCliftDialect
         MLIRCliftUtils)
