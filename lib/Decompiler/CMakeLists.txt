revng_add_analyses_library(Decompiler revngc
                           ASTBuildAnalysis.cpp
                           CDecompilerAction.cpp
                           CDecompilerBeautify.cpp
                           CDecompilerPass.cpp
                           DecompilationHelpers.cpp
                           FuncDeclCreationAction.cpp
                           GlobalDeclCreationAction.cpp
                           IRASTTypeTranslation.cpp
                           MarkForSerialization.cpp)

target_link_libraries(Decompiler
                      revng::revngSupport
                      EnforceCFGCombingPass
                      clangTooling
                      clangAST
                      clangBasic
                      clangFrontend
                      ${LLVM_LIBRARIES})
