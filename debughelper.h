#ifndef _DEBUGHELPER_H
#define _DEBUGHELPER_H

// Standard includes
#include <memory>
#include <ostream>
#include <string>

// LLVM includes
#include "llvm/IR/DIBuilder.h"

// Local includes
#include "revamb.h"

namespace llvm {
class DIBuilder;
class Module;
class DICompileUnit;
class DISubprogram;
class Function;
}

/// AssemblyAnnotationWriter implementation inserting in the generated LLVM IR
/// comments containing the original assembly and the PTC. It can also decorate
/// the IR with debug information (i.e. DILocations) refered to the generated
/// LLVM IR itself.
class DebugAnnotationWriter : public llvm::AssemblyAnnotationWriter {
public:
  DebugAnnotationWriter(llvm::LLVMContext& Context,
                        llvm::Metadata *Scope,
                        bool DebugInfo);

  virtual void emitInstructionAnnot(const llvm::Instruction *TheInstruction,
                                    llvm::formatted_raw_ostream &Output);

private:
  llvm::LLVMContext &Context;
  llvm::Metadata *Scope;
  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
  bool DebugInfo;
};

/// Handle all the debug-related operations of code generation
class DebugHelper {
public:
  DebugHelper(std::string Output,
              std::string Debug,
              llvm::Module *TheModule,
              DebugInfoType Type);

  /// \brief Handle a new function
  ///
  /// Generates the debug information for the given function and caches it for
  /// future use.
  void newFunction(llvm::Function *Function);

  /// Decorates the current function with the request debug info
  void generateDebugInfo();

  /// Serializes to the given stream the module, with or without debug info
  void print(std::ostream& Output, bool DebugInfo);

  /// Copy the debug file to the output path, if they are the same
  bool copySource();

private:
  /// Create a new AssemblyAnnotationWriter
  ///
  /// \param DebugInfo whether to decorate the IR with debug information or not
  DebugAnnotationWriter *annotator(bool DebugInfo);

private:
  std::string OutputPath;
  std::string DebugPath;
  llvm::DIBuilder Builder;
  DebugInfoType Type;
  llvm::Module *TheModule;
  llvm::DICompileUnit *CompileUnit;
  llvm::DISubprogram *CurrentSubprogram;
  llvm::Function *CurrentFunction;
  std::unique_ptr<DebugAnnotationWriter> Annotator;

  unsigned OriginalInstrMDKind;
  unsigned PTCInstrMDKind;
  unsigned DbgMDKind;
};

#endif // _DEBUGHELPER_H
