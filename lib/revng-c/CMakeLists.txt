set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(revng-c SHARED
            DecompilationPass.cpp
            ASTBuildAnalysis.cpp
            DecompilationAction.cpp
            DecompilationHelpers.cpp
            FuncDeclCreationAction.cpp
            GlobalDeclCreationAction.cpp
            IRASTTypeTranslation.cpp
            )

target_link_libraries(revng-c
                      ${REVNG_SUPPORT_LIBRARY}
                      clangTooling
                      clangAST
                      clangBasic
                      clangFrontend
                      ${LLVM_LIBRARIES}
                      )

install(TARGETS revng-c
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
