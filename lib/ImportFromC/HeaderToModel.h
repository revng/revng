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

namespace revng {
struct ParseCCodeError {
  std::string ErrorMessage;
  unsigned Line = 0;
  unsigned Column = 0;
};
} // namespace revng

namespace clang {

class CompilerInstance;

namespace tooling {

class HeaderToModelDiagnosticConsumer : public clang::DiagnosticConsumer {
public:
  HeaderToModelDiagnosticConsumer(clang::DiagnosticsEngine &D) :
    Client(D.getClient()), ClientOwner(D.takeClient()) {}

  void EndSourceFile() override;

  void HandleDiagnostic(clang::DiagnosticsEngine::Level DiagLevel,
                        const clang::Diagnostic &Info) override;

  std::optional<revng::ParseCCodeError> &getError() { return Error; }

private:
  clang::DiagnosticConsumer *Client;
  std::unique_ptr<clang::DiagnosticConsumer> ClientOwner;
  std::optional<revng::ParseCCodeError> Error;
};

class HeaderToModelAction : public ASTFrontendAction {
protected:
  HeaderToModelAction(TupleTree<model::Binary> &Model,
                      enum ImportFromCOption AnalysisOption,
                      std::optional<revng::ParseCCodeError> &Error) :
    Model(Model), AnalysisOption(AnalysisOption), Error(Error) {}

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

  // This holds the error message reported by clang or revng, if any.
  std::optional<revng::ParseCCodeError> &Error;

private:
  HeaderToModelDiagnosticConsumer *DiagConsumer = nullptr;
};

// Handle Edit Type option.
class HeaderToModelEditTypeAction : public HeaderToModelAction {
public:
  HeaderToModelEditTypeAction(TupleTree<model::Binary> &Model,
                              std::optional<revng::ParseCCodeError> &Error,
                              std::optional<model::Type *> &Type) :
    HeaderToModelAction(Model, ImportFromCOption::EditType, Error),
    Type(Type) {}

private:
  // Type to be edited.
  std::optional<model::Type *> &Type;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Edit Function Prototype option.
class HeaderToModelEditFunctionAction : public HeaderToModelAction {
public:
  HeaderToModelEditFunctionAction(TupleTree<model::Binary> &Model,
                                  std::optional<revng::ParseCCodeError> &Error,
                                  std::optional<model::Function> &Function) :
    HeaderToModelAction(Model, ImportFromCOption::EditFunctionPrototype, Error),
    Function(Function) {}

private:
  // Function to be edited if we are editing function prototype.
  std::optional<model::Function> &Function;

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

// Handle Add Type option.
class HeaderToModelAddTypeAction : public HeaderToModelAction {
public:
  HeaderToModelAddTypeAction(TupleTree<model::Binary> &Model,
                             std::optional<revng::ParseCCodeError> &Error) :
    HeaderToModelAction(Model, ImportFromCOption::AddType, Error) {}

public:
  virtual std::unique_ptr<ASTConsumer> newASTConsumer() override;
};

} // end namespace tooling
} // namespace clang
