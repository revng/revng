revng_add_analyses_library_internal(
  revngMLIRPipes
  NameInjectionPipe.cpp
  CliftContainer.cpp
  ImportCliftTypesPipe.cpp
  CBackendPipe.cpp
  LLVMToCliftPipe.cpp
  ModelVerifyPipe.cpp
  Legalization.cpp)

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
