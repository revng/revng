#include <string>

int Translate(std::string OutputPath,
              llvm::ArrayRef<uint8_t> Code,
              DebugInfoType DebugInfo,
              std::string DebugPath);

