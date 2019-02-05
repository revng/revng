#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <revng/Support/Assert.h>

#include "DecompilationAction.h"

#include "ASTBuildAnalysis.h"
#include "FuncDeclCreationAction.h"
#include "GlobalDeclCreationAction.h"
#include "IRASTTypeTranslation.h"

namespace clang {
namespace tooling {

using GlobalsMap = GlobalDeclCreationAction::GlobalsMap;
using FunctionsMap = FuncDeclCreationAction::FunctionsMap;

static void createLocalVarDecls(FunctionsMap::value_type &FPair,
                                IR2AST::SerializationInfo &ASTInfo) {
  llvm::Function &F = *FPair.first;
  clang::FunctionDecl *FDecl = FPair.second;
  ASTContext &ASTCtx = FDecl->getASTContext();
  SmallVector<clang::Decl *, 16> LocalVarDecls;
  for (auto &DeclPair : ASTInfo.AllocaDecls)
    LocalVarDecls.push_back(DeclPair.second);
  for (auto &DeclPair : ASTInfo.VarDecls)
    LocalVarDecls.push_back(DeclPair.second);

  // Body is a CompoundStmt, composed of:
  // - an (optional) initial sequence of DeclStmts holding the declaration of
  //   the local variables, if any.
  // - a CompoundStmt for each BasicBlock, if any.
  unsigned NumLocalVars = LocalVarDecls.size();
  unsigned NumStmtsInBody = F.size() + NumLocalVars;
  CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, NumStmtsInBody);
  FDecl->setBody(Body);

  for (unsigned I = 0; I < NumLocalVars; ++I) {
    Decl *VDecl = LocalVarDecls[I];
    auto *LocalVarDeclStmt = new (ASTCtx) DeclStmt(DeclGroupRef(VDecl), {}, {});
    Body->body_begin()[I] = LocalVarDeclStmt;
  }

  int I = NumLocalVars;
  auto End = ASTInfo.InstrStmts.end();
  for (llvm::BasicBlock &BB : F) {
    SmallVector<clang::Stmt *, 16> BBStmts;
    for (llvm::Instruction &Instr : BB) {
      auto It = ASTInfo.InstrStmts.find(&Instr);
      if (It != End)
        BBStmts.push_back(It->second);
    }
    auto *BBCompoundStmt = CompoundStmt::Create(ASTCtx, BBStmts, {}, {});
    Body->body_begin()[I] = new (ASTCtx)
      LabelStmt({}, ASTInfo.LabelDecls.at(&BB), BBCompoundStmt);
    ++I;
  }
}

static void createInstructionStmts(FunctionsMap::value_type & /*FPair*/,
                                   IR2AST::SerializationInfo & /*ASTInfo*/) {
}

static void buildFunctionBody(FunctionsMap::value_type &FPair,
                              IR2AST::SerializationInfo &ASTInfo) {
  createLocalVarDecls(FPair, ASTInfo);
  createInstructionStmts(FPair, ASTInfo);
}

class Decompiler : public ASTConsumer {
public:
  explicit Decompiler(llvm::Module &M,
                      const llvm::Function *F,
                      std::unique_ptr<llvm::raw_ostream> Out) :
    M(M),
    TheF(F),
    Out(std::move(Out)) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override {
    using ConsumerPtr = std::unique_ptr<ASTConsumer>;

    // Build declaration of global variables
    ConsumerPtr GlobalDeclCreation = CreateGlobalDeclCreator(M, GlobalVarAST);
    GlobalDeclCreation->HandleTranslationUnit(Context);
    // Build function declaration
    ConsumerPtr FunDeclCreation = CreateFuncDeclCreator(M,
                                                        FunctionDecls,
                                                        FunctionDefs);
    FunDeclCreation->HandleTranslationUnit(Context);

    for (auto &F : FunctionDefs) {
      llvm::Function *LLVMFunc = F.first;
      revng_assert(not LLVMFunc->isDeclaration());
      const llvm::StringRef FName = LLVMFunc->getName();
      revng_assert(FName.startswith("bb."));

      std::map<const llvm::BasicBlock *, clang::CompoundStmt *> BBAST;

      IR2AST::Analysis IR2ASTBuildAnalysis(*LLVMFunc,
                                           Context,
                                           *F.second,
                                           GlobalVarAST,
                                           FunctionDecls);
      IR2ASTBuildAnalysis.initialize();
      IR2ASTBuildAnalysis.run();

      if (TheF == nullptr or LLVMFunc == TheF) {
        auto &&ASTInfo = IR2ASTBuildAnalysis.extractASTInfo();
        buildFunctionBody(F, ASTInfo);
      }
    }

    // ConsumerPtr Dumper = CreateASTDumper(nullptr, "", true, false, false);
    // Dumper->HandleTranslationUnit(Context);
    ConsumerPtr Printer = CreateASTPrinter(std::move(Out), "");
    Printer->HandleTranslationUnit(Context);
  }

private:
  llvm::Module &M;
  const llvm::Function *TheF;
  std::unique_ptr<llvm::raw_ostream> Out;
  FunctionsMap FunctionDecls;
  FunctionsMap FunctionDefs;
  GlobalsMap GlobalVarAST;
};

std::unique_ptr<ASTConsumer> DecompilationAction::newASTConsumer() {
  return std::make_unique<Decompiler>(M, F, std::move(O));
}

} // end namespace tooling
} // end namespace clang
