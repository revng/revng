revng_add_analyses_library(revngcMLIRPipes revngc MLIRPipe.cpp)

target_link_libraries(
  revngcMLIRPipes
  PUBLIC revng::revngPipeline
         revng::revngPipes
         MLIRTransforms
         MLIRDialect
         MLIRIR
         MLIRLLVMDialect
         MLIRDLTIDialect
         MLIRLLVMIRToLLVMTranslation)
