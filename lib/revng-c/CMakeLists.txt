revng_add_analyses_library(revng-c
                           revngc
                           DecompilationPass.cpp
                           ASTBuildAnalysis.cpp
                           DecompilationAction.cpp
                           DecompilationHelpers.cpp
                           FuncDeclCreationAction.cpp
                           GlobalDeclCreationAction.cpp
                           IRASTTypeTranslation.cpp)

target_link_libraries(revng-c
                      revng::revngSupport
                      EnforceCFGCombingPass
                      clangTooling
                      clangAST
                      clangBasic
                      clangFrontend
                      ${LLVM_LIBRARIES})
