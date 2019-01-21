set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(revng-c
                           DecompilationPass.cpp
                           ASTBuildAnalysis.cpp
                           DecompilationAction.cpp
                           DecompilationHelpers.cpp
                           FuncDeclCreationAction.cpp
                           GlobalDeclCreationAction.cpp
                           IRASTTypeTranslation.cpp
                           )

target_link_libraries(revng-c
                      revng::revngSupport
                      clangTooling
                      clangAST
                      clangBasic
                      clangFrontend
                      ${LLVM_LIBRARIES}
                      )
