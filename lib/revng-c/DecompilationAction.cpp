#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <clang/AST/Stmt.h>
#include <clang/Basic/SourceLocation.h>

#include <revng/Support/Assert.h>

#include "revng-c/RestructureCFGPass/ASTTree.h"

#include "DecompilationAction.h"

#include "ASTBuildAnalysis.h"
#include "FuncDeclCreationAction.h"
#include "GlobalDeclCreationAction.h"
#include "IRASTTypeTranslation.h"

namespace clang {
namespace tooling {

using GlobalsMap = GlobalDeclCreationAction::GlobalsMap;
using FunctionsMap = FuncDeclCreationAction::FunctionsMap;

static clang::Stmt *buildScope(ASTNode *N,
                               IR2AST::StmtMap &InstrStmts,
                               GlobalsMap &GlobalVarAST,
                               FunctionsMap &FunctionAST,
                               clang::ASTContext &ASTCtx,
                               IR2AST::SerializationInfo &ASTInfo) {
  if (N == nullptr)
    return nullptr;
  switch (N->getKind()) {
    case ASTNode::NodeKind::NK_Break:
      return new (ASTCtx) clang::BreakStmt(SourceLocation{});
    case ASTNode::NodeKind::NK_Continue:
      return new (ASTCtx) clang::ContinueStmt(SourceLocation{});
    case ASTNode::NodeKind::NK_Code: {
      SmallVector<clang::Stmt *, 16> Stmts;
      CodeNode *Code = cast<CodeNode>(N);
      llvm::BasicBlock *BB = Code->getOriginalBB();
      revng_assert(BB != nullptr);
      auto End = InstrStmts.end();
      for (llvm::Instruction &Instr : *BB) {
        auto It = InstrStmts.find(&Instr);
        if (It != End)
          Stmts.push_back(It->second);
      }
      return CompoundStmt::Create(ASTCtx, Stmts, {}, {});
    }
    case ASTNode::NodeKind::NK_If: {
      IfNode *If = cast<IfNode>(N);
      clang::Stmt *ThenScope = buildScope(If->getThen(), InstrStmts, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
      clang::Stmt *ElseScope = buildScope(If->getElse(), InstrStmts, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);
      llvm::BasicBlock *CondBlock = If->getUniqueCondBlock();

      SmallVector<clang::Stmt *, 16> Stmts;

      auto End = InstrStmts.end();
      for (llvm::Instruction &Instr : *CondBlock) {
        auto It = InstrStmts.find(&Instr);
        if (It != End)
          Stmts.push_back(It->second);
      }

      llvm::Instruction *CondTerminator = CondBlock->getTerminator();
      llvm::BranchInst *Br = cast<llvm::BranchInst>(CondTerminator);
      revng_assert(Br->isConditional());
      llvm::Value *CondValue = Br->getCondition();
      clang::Expr *CondExpr = getExprForValue(CondValue, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);

      Stmts.push_back(new (ASTCtx) IfStmt (ASTCtx, {}, false, nullptr, nullptr,
                            CondExpr, ThenScope, {}, ElseScope));
      return CompoundStmt::Create(ASTCtx, Stmts, {}, {});
    }
    case ASTNode::NodeKind::NK_Scs: {
      ScsNode *LoopBody = cast<ScsNode>(N);
      clang::Stmt *Body = buildScope(LoopBody->getBody(), InstrStmts, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo);

      QualType UInt = ASTCtx.UnsignedIntTy;
      uint64_t UIntSize = ASTCtx.getTypeSize(UInt);
      clang::Expr *TrueCond = IntegerLiteral::Create(ASTCtx,
                                                     llvm::APInt(UIntSize, 1),
                                                     UInt,
                                                     {});

      return new (ASTCtx) WhileStmt(ASTCtx, nullptr, TrueCond, Body, {});
    }
    case ASTNode::NodeKind::NK_List: {
      SmallVector<clang::Stmt *, 16> Stmts;
      SequenceNode *Seq = cast<SequenceNode>(N);
      for (ASTNode *Child : Seq->nodes()) {
        if (clang::Stmt *S = buildScope(Child, InstrStmts, GlobalVarAST, FunctionAST, ASTCtx, ASTInfo))
          Stmts.push_back(S);
      }
      return CompoundStmt::Create(ASTCtx, Stmts, {}, {});
    }
    default:
      revng_abort();
      return nullptr;
  }
}

static void buildFunctionBody(FunctionsMap::value_type &FPair,
                              ASTTree &CombedAST,
                              GlobalsMap &GlobalVarAST,
                              FunctionsMap &FunctionAST,
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
  unsigned NumStmtsInBody = 1 + NumLocalVars;
  CompoundStmt *Body = CompoundStmt::CreateEmpty(ASTCtx, NumStmtsInBody);
  FDecl->setBody(Body);

  for (unsigned I = 0; I < NumLocalVars; ++I) {
    Decl *VDecl = LocalVarDecls[I];
    auto *LocalVarDeclStmt = new (ASTCtx) DeclStmt(DeclGroupRef(VDecl), {}, {});
    Body->body_begin()[I] = LocalVarDeclStmt;
  }

  Body->body_begin()[NumLocalVars] =
  buildScope(CombedAST.getRoot(), ASTInfo.InstrStmts,
  GlobalVarAST,
  FunctionAST,
  ASTCtx,
  ASTInfo);

#if 0
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
#endif
}

class Decompiler : public ASTConsumer {
public:
  explicit Decompiler(llvm::Function &F,
                      ASTTree &CombedAST,
                      std::unique_ptr<llvm::raw_ostream> Out) :
    TheF(F),
    CombedAST(CombedAST),
    Out(std::move(Out)) {}

  virtual void HandleTranslationUnit(ASTContext &Context) override {
    using ConsumerPtr = std::unique_ptr<ASTConsumer>;

    llvm::Module *M = TheF.getParent();

    // Build declaration of global variables
    ConsumerPtr GlobalDeclCreation = CreateGlobalDeclCreator(*M, GlobalVarAST);
    GlobalDeclCreation->HandleTranslationUnit(Context);
    // Build function declaration
    ConsumerPtr FunDeclCreation = CreateFuncDeclCreator(*M,
                                                        FunctionDecls,
                                                        FunctionDefs);
    FunDeclCreation->HandleTranslationUnit(Context);

    revng_assert(not TheF.isDeclaration());
    revng_assert(TheF.getName().startswith("bb."));
    auto It = FunctionDefs.find(&TheF);
    revng_assert(It != FunctionDefs.end());
    clang::FunctionDecl *FunctionDecl = It->second;

    IR2AST::Analysis IR2ASTBuildAnalysis(TheF,
                                         Context,
                                         *FunctionDecl,
                                         GlobalVarAST,
                                         FunctionDecls);
    IR2ASTBuildAnalysis.initialize();
    IR2ASTBuildAnalysis.run();
    auto &&ASTInfo = IR2ASTBuildAnalysis.extractASTInfo();

    buildFunctionBody(*It, CombedAST, GlobalVarAST, FunctionDecls, ASTInfo);

    for (auto &F : FunctionDefs) {
      llvm::Function *LLVMFunc = F.first;
      const llvm::StringRef FName = LLVMFunc->getName();

    }

    // ConsumerPtr Dumper = CreateASTDumper(nullptr, "", true, false, false);
    // Dumper->HandleTranslationUnit(Context);
    ConsumerPtr Printer = CreateASTPrinter(std::move(Out), "");
    Printer->HandleTranslationUnit(Context);
  }

private:
  llvm::Function &TheF;
  ASTTree &CombedAST;
  std::unique_ptr<llvm::raw_ostream> Out;
  FunctionsMap FunctionDecls;
  FunctionsMap FunctionDefs;
  GlobalsMap GlobalVarAST;
};

std::unique_ptr<ASTConsumer> DecompilationAction::newASTConsumer() {
  return std::make_unique<Decompiler>(F, CombedAST, std::move(O));
}

} // end namespace tooling
} // end namespace clang
