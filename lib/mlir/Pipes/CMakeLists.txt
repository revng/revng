revng_add_analyses_library_internal(revngMLIRPipes MLIRContainer.cpp
                                    ImportCliftTypesPipe.cpp)

target_link_libraries(
  revngMLIRPipes
  PUBLIC revngPipeline
         revngPipes
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
