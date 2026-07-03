#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendAction.h"

#include "revng/Model/Binary.h"
#include "revng/Model/ForwardDecls.h"
#include "revng/TupleTree/TupleTree.h"

#include "ImportFromCAnalysis.h"

using ImportingErrorList = std::vector<std::string>;

namespace clang {

class CompilerInstance;

namespace tooling {

class ImportFromCDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  ImportFromCDiagnosticConsumer(clang::DiagnosticsEngine &D) :
    Client(D.getClient()), ClientOwner(D.takeClient()) {}

  void EndSourceFile() override;

  void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;

  ImportingErrorList &&extractErrors() { return std::move(Errors); }

private:
  clang::DiagnosticConsumer *Client;
  std::unique_ptr<clang::DiagnosticConsumer> ClientOwner;
  ImportingErrorList Errors;
};

class ImportFromCAction : public ASTFrontendAction {
protected:
  ImportFromCAction(TupleTree<model::Binary> &Model,
                    enum ImportFromCOption AnalysisOption,
                    ImportingErrorList &Errors) :
    Model(Model), AnalysisOption(AnalysisOption), Errors(Errors) {}

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() = 0;

  virtual std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &, llvm::StringRef) override;

  virtual bool BeginInvocation(clang::CompilerInstance &CI) override;

  virtual void EndSourceFile() override;

protected:
  TupleTree<model::Binary> &Model;

  // This indiacates which feature is used (edit/add type, edit function
  // prototype).
  enum ImportFromCOption AnalysisOption;

  // This holds error messages reported by clang or revng, if any.
  ImportingErrorList &Errors;

private:
  ImportFromCDiagnosticConsumer *DiagConsumer = nullptr;
};

// Handle Edit Type option.
class ImportFromCEditTypeAction : public ImportFromCAction {
public:
  ImportFromCEditTypeAction(TupleTree<model::Binary> &Model,
                            ImportingErrorList &Errors,
                            std::optional<model::TypeDefinition::Key> Type) :
    ImportFromCAction(Model, ImportFromCOption::EditType, Errors), Type(Type) {}

private:
  // Type to be edited.
  std::optional<model::TypeDefinition::Key> Type;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Edit Function Prototype option.
class ImportFromCEditFunctionAction : public ImportFromCAction {
public:
  ImportFromCEditFunctionAction(TupleTree<model::Binary> &Model,
                                ImportingErrorList &Errors,
                                MetaAddress FunctionEntry) :
    ImportFromCAction(Model, ImportFromCOption::EditFunctionPrototype, Errors),
    FunctionEntry(FunctionEntry) {}

private:
  // Function to be edited if we are editing function prototype.
  MetaAddress FunctionEntry;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Add Type option.
class ImportFromCAddTypeAction : public ImportFromCAction {
public:
  ImportFromCAddTypeAction(TupleTree<model::Binary> &Model,
                           ImportingErrorList &Errors) :
    ImportFromCAction(Model, ImportFromCOption::AddType, Errors) {}

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

} // end namespace tooling
} // namespace clang
