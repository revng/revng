#ifndef _DEBUGHELPER_H
#define _DEBUGHELPER_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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

/// \brief AssemblyAnnotationWriter decorating the output withe debug
///        information
///
/// AssemblyAnnotationWriter implementation inserting in the generated LLVM IR
/// comments containing the original assembly and the PTC. It can also decorate
/// the IR with debug information (i.e. DILocations) refered to the generated
/// LLVM IR itself.
class DebugAnnotationWriter : public llvm::AssemblyAnnotationWriter {
public:
  /// \brief Create a new DebugAnnotationWriter
  ///
  /// \warning If \p DebugInfo is `true`, the produced output should be
  ///          discarded since it will contain errors. \p DebugInfo should be
  ///          set to `true` in a first run to produce the metadata, and then a
  ///          new DebugAnnotationWriter with `DebugInfo = false` should be
  ///          created and run to produce an output without errors.
  ///
  /// \param Context the LLVM context.
  /// \param Scope the scope, typically a `DISubprogram`.
  /// \param DebugInfo whether to decorate the IR being serialized with debug
  ///        metadata refering to the produce IR itself or not.
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

/// \brief Handle printing the IR in textual form, possibly with debug
///        information
class DebugHelper {
public:
  /// \brief Create a new DebugHelper
  ///
  /// \param Output path where the LLVM IR should be stored.
  /// \param Debug path where the debug output should be stored. If empty, \p
  ///        Output will be used along with a suffix, e.g. `.pts` if \p Type is
  ///        DebugInfoType::PTC, `.S` if it's DebugInfoType::OriginalAssembly or
  ///        will match \p Output if \p Type is DebugInfoType::LLVMIR.
  /// \param TheModule the LLVM module to print out.
  /// \param Type type of debug information requested.
  DebugHelper(std::string Output,
              std::string Debug,
              llvm::Module *TheModule,
              DebugInfoType Type);

  /// \brief Handle a new function
  ///
  /// Generates the debug information for the given function and caches it for
  /// future use.
  void newFunction(llvm::Function *Function);

  /// Decorates the current function with the requested debug info
  void generateDebugInfo();

  /// Serializes to the given stream the module, with or without debug info
  void print(std::ostream& Output, bool DebugInfo);

  /// Copy the debug file to the output path, if they are the same
  bool copySource();

private:
  /// Create a new AssemblyAnnotationWriter
  ///
  /// \param DebugInfo whether to create an annotator producing with debug
  ///        information referred to itself or not.
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
