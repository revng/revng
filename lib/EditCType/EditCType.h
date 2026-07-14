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

#include "EditCTypeAnalysis.h"

using ImportingErrorList = std::vector<std::string>;

namespace clang {

class CompilerInstance;

namespace tooling {

class EditCTypeDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  EditCTypeDiagnosticConsumer(clang::DiagnosticsEngine &D) :
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

class EditCTypeAction : public ASTFrontendAction {
protected:
  EditCTypeAction(TupleTree<model::Binary> &Model,
                  enum EditCTypeOption AnalysisOption,
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
  enum EditCTypeOption AnalysisOption;

  // This holds error messages reported by clang or revng, if any.
  ImportingErrorList &Errors;

private:
  EditCTypeDiagnosticConsumer *DiagConsumer = nullptr;
};

// Handle Edit Type option.
class EditCTypeEditTypeAction : public EditCTypeAction {
public:
  EditCTypeEditTypeAction(TupleTree<model::Binary> &Model,
                          ImportingErrorList &Errors,
                          std::optional<model::TypeDefinition::Key> Type) :
    EditCTypeAction(Model, EditCTypeOption::EditType, Errors), Type(Type) {}

private:
  // Type to be edited.
  std::optional<model::TypeDefinition::Key> Type;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Edit Function Prototype option.
class EditCTypeEditFunctionAction : public EditCTypeAction {
public:
  EditCTypeEditFunctionAction(TupleTree<model::Binary> &Model,
                              ImportingErrorList &Errors,
                              MetaAddress FunctionEntry) :
    EditCTypeAction(Model, EditCTypeOption::EditFunctionPrototype, Errors),
    FunctionEntry(FunctionEntry) {}

private:
  // Function to be edited if we are editing function prototype.
  MetaAddress FunctionEntry;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Add Type option.
class EditCTypeAddTypeAction : public EditCTypeAction {
public:
  EditCTypeAddTypeAction(TupleTree<model::Binary> &Model,
                         ImportingErrorList &Errors) :
    EditCTypeAction(Model, EditCTypeOption::AddType, Errors) {}

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

} // end namespace tooling
} // namespace clang
