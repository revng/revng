#ifndef _PTCTOLLVMIR_H
#define _PTCTOLLVMIR_H

#include <cstdint>
#include <string>
#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "revamb.h"

namespace llvm {
class LLVMContext;
class Function;
class Module;
};

class DebugHelper;

/// Translates code from an architecture to LLVM IR.
class CodeGenerator {
public:
  /// Create a new code generator translating code from an architecture to
  /// another, writing the corresponding LLVM IR and debug source file of the
  /// requested type to file.
  ///
  /// \param Source source architecture.
  /// \param Target target architecture.
  /// \param OutputPath path where the generate LLVM IR must be saved.
  /// \param DebugInfo type of debug information to generate.
  /// \param DebugPath path where the debugging source file must be written.
  CodeGenerator(Architecture& Source,
                Architecture& Target,
                std::string OutputPath,
                DebugInfoType DebugInfo,
                std::string DebugPath);

  ~CodeGenerator();

  /// \brief Creates an LLVM function for the code in the specified memory area.
  /// If debug information have been requested, the debug source files will be
  /// create in this phase.
  ///
  /// \param Name the name to give to the newly created function.
  /// \param Code reference to memory area containing the code to translate.
  int translate(llvm::ArrayRef<uint8_t> Code, std::string Name);

  /// Serialize the generated LLVM IR to the specified output path.
  void serialize();
private:
  Architecture& SourceArchitecture;
  Architecture& TargetArchitecture;
  llvm::LLVMContext& Context;
  std::unique_ptr<llvm::Module> TheModule;
  std::unique_ptr<DebugHelper> Debug;
  std::string OutputPath;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
};

#endif // _PTCTOLLVMIR_H
