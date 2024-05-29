revng_add_analyses_library(revngcMLIRPipes revngc MLIRContainer.cpp
                           ImportLLVMToMLIRPipe.cpp ImportCliftTypesPipe.cpp)

target_link_libraries(
  revngcMLIRPipes
  PUBLIC revng::revngPipeline
         revng::revngPipes
         MLIRTransforms
         MLIRDialect
         MLIRIR
         MLIRLLVMDialect
         MLIRDLTIDialect
         MLIRLLVMIRToLLVMTranslation
         MLIRBytecodeReader
         MLIRBytecodeWriter
         MLIRCliftDialect
         MLIRCliftUtils)
