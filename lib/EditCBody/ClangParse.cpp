//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ClangToModel/CompileFlags.h"
#include "revng/ClangToModel/QualTypeToModel.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"

#include "ClangParse.h"
#include "Statements.h"

using namespace llvm;
using namespace revng::editcbody;

static Logger Log("edit-c-body-clang");

namespace {

//
// Comment parsing
//

/// Strip the `//` or `/* */` markers off a raw comment, returning its body.
///
/// \note We ignore doxygen (`///`) comments.
std::string commentBody(StringRef Raw) {
  Raw = Raw.trim();
  if (Raw.starts_with("//"))
    return Raw.drop_front(2).trim().str();
  if (Raw.starts_with("/*")) {
    Raw = Raw.drop_front(2);
    if (Raw.ends_with("*/"))
      Raw = Raw.drop_back(2);
    return Raw.trim().str();
  }
  return Raw.trim().str();
}

/// Split a raw comment into its individual body lines and append the non-blank
/// ones to `Lines`, stripped of their markers.
///
/// Clang merges a run of consecutive `//` lines into one comment, so a stacked
/// `// RENAME:` / `// RETYPE:` pair reaches us as a single comment with an
/// embedded newline. Splitting on `\n` here recovers the separate lines, so
/// each can be classified on its own.
void appendCommentLines(StringRef Raw,
                        llvm::SmallVectorImpl<std::string> &Lines) {
  llvm::SmallVector<StringRef> Split;
  Raw.split(Split, '\n');
  for (StringRef Line : Split) {
    std::string Body = commentBody(Line);
    if (not Body.empty())
      Lines.push_back(std::move(Body));
  }
}

/// Whether the comment starting at `Begin` is the first thing on its line, i.e.
/// only whitespace precedes it. A leading comment must start its own line; a
/// comment trailing a statement on the same line is not attached to any
/// statement and is ignored.
bool startsOwnLine(StringRef Buffer, unsigned Begin) {
  size_t LineStart = Buffer.substr(0, Begin).rfind('\n');
  LineStart = (LineStart == StringRef::npos) ? 0 : LineStart + 1;
  return llvm::all_of(Buffer.substr(LineStart, Begin - LineStart),
                      llvm::isSpace);
}

//
// RETYPE type resolution
//

/// Prefix shared by the synthetic probe declarations (`__revng_retype_<index>`)
/// used to resolve the types named by `RETYPE:` directives during the parse.
constexpr llvm::StringRef RetypeProbePrefix = "__revng_retype_";

/// Collect the type named by each `RETYPE:` directive in the code, in order of
/// first appearance and without duplicates. This runs before Clang, so the
/// types can be turned into synthetic typedefs and resolved during the parse.
/// A superset of the directives the analysis later acts on is fine: extra
/// entries are simply never looked up.
std::vector<std::string> collectRetypeStrings(StringRef CCode) {
  std::vector<std::string> Result;
  llvm::SmallVector<StringRef> Lines;
  CCode.split(Lines, '\n');
  for (StringRef Line : Lines) {
    StringRef Directive = StringRef(commentBody(Line)).trim();
    if (Directive.consume_front("RETYPE:")) {
      std::string Type = Directive.trim().str();
      if (not Type.empty() and not llvm::is_contained(Result, Type))
        Result.push_back(std::move(Type));
    }
  }
  return Result;
}

/// Render `void __revng_retype_<index>(<type>);`. A function parameter is a
/// C type-name position, so the type is written verbatim, whatever its shape
/// (arrays, pointers to arrays, ...); no declarator surgery is needed. The
/// parameter's un-decayed type is read back after parsing.
std::string makeRetypeProbe(StringRef TypeString, size_t Index) {
  return (llvm::Twine("void ") + RetypeProbePrefix + llvm::Twine(Index) + "("
          + TypeString.trim() + ");\n")
    .str();
}

//
// Clang statement walk
//

struct ParseOutput {
  std::vector<CStatement> Statements;
  std::map<std::string, model::UpcastableType> ResolvedTypes;
  std::string Error;
};

/// Consumer that finds the single function definition in the main file and
/// flattens its body into an ordered list of statements, attaching to each the
/// comments the user placed on the line(s) before it.
class ParseConsumer : public clang::ASTConsumer {
private:
  ParseOutput &Output;
  const model::Binary &Binary;
  llvm::ArrayRef<std::string> RetypeStrings;
  clang::SourceManager *Sources = nullptr;

public:
  ParseConsumer(ParseOutput &Output,
                const model::Binary &Binary,
                llvm::ArrayRef<std::string> RetypeStrings) :
    Output(Output), Binary(Binary), RetypeStrings(RetypeStrings) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Sources = &Context.getSourceManager();
    clang::FileID MainFile = Sources->getMainFileID();

    const clang::FunctionDecl *Target = nullptr;
    for (const clang::Decl *Declaration :
         Context.getTranslationUnitDecl()->decls()) {
      auto *Function = clang::dyn_cast<clang::FunctionDecl>(Declaration);
      if (Function == nullptr)
        continue;

      if (Function->getIdentifier()
          and Function->getName().starts_with(RetypeProbePrefix)) {
        resolveRetype(*Function, Context);
        continue;
      }

      if (not Function->isThisDeclarationADefinition())
        continue;

      if (not Sources->isInMainFile(Function->getLocation()))
        continue;

      Target = Function;
    }

    if (Target == nullptr or Target->getBody() == nullptr) {
      Output.Error = "no function definition found in the provided C code";
      return;
    }

    flattenStatement(Target->getBody());

    if (not Output.Error.empty())
      return;

    attachComments(Context, MainFile);
  }

private:
  /// A synthetic `void __revng_retype_N(<type>);` carries the type named by the
  /// N-th `RETYPE:` directive as its single parameter. Resolve it to a model
  /// type, keyed by the directive text so the analysis can look it up.
  void resolveRetype(const clang::FunctionDecl &Probe,
                     clang::ASTContext &Context) {
    StringRef Name = Probe.getName();
    if (not Name.consume_front(RetypeProbePrefix))
      return;

    unsigned Index = 0;
    if (Name.getAsInteger(10, Index) or Index >= RetypeStrings.size())
      return;
    if (Probe.getNumParams() != 1)
      return;

    std::vector<std::string> Ignored;
    clang::QualType Argument = Probe.getParamDecl(0)->getOriginalType();
    model::UpcastableType Resolved = revng::qualTypeToModel(Argument,
                                                            Binary,
                                                            Context,
                                                            Ignored,
                                                            "edit-c-body:");
    Output.ResolvedTypes[RetypeStrings[Index]] = std::move(Resolved);
  }

  void push(StatementKind Kind, const clang::Stmt *Statement) {
    auto BeginLocation = Sources->getExpansionLoc(Statement->getBeginLoc());
    unsigned Offset = Sources->getFileOffset(BeginLocation);
    Output.Statements.push_back({ Kind, Offset, {} });
  }

  /// Flatten a statement into the ordered list, descending into nested bodies.
  /// Blocks are transparent, so a compound statement yields its children
  /// directly, without producing a node of its own.
  RecursiveCoroutine<void> flattenStatement(const clang::Stmt *Statement) {
    using namespace clang;

    if (Statement == nullptr or isa<NullStmt>(Statement)) {
      rc_return;
    } else if (auto *Compound = dyn_cast<CompoundStmt>(Statement)) {
      for (const Stmt *Child : Compound->body())
        rc_recur flattenStatement(Child);
    } else if (auto *Label = dyn_cast<LabelStmt>(Statement)) {
      // Labels and cases wrap their sub-statement in the Clang AST, whereas in
      // Clift they are siblings. Flatten them out.
      push(StatementKind::Label, Statement);
      rc_recur flattenStatement(Label->getSubStmt());
    } else if (auto *Case = dyn_cast<CaseStmt>(Statement)) {
      push(StatementKind::Case, Statement);
      rc_recur flattenStatement(Case->getSubStmt());
    } else if (auto *Default = dyn_cast<DefaultStmt>(Statement)) {
      push(StatementKind::Default, Statement);
      rc_recur flattenStatement(Default->getSubStmt());
    } else if (auto *If = dyn_cast<IfStmt>(Statement)) {
      push(StatementKind::If, Statement);
      rc_recur flattenStatement(If->getThen());
      if (If->getElse() != nullptr)
        rc_recur flattenStatement(If->getElse());
    } else if (auto *While = dyn_cast<WhileStmt>(Statement)) {
      push(StatementKind::While, Statement);
      rc_recur flattenStatement(While->getBody());
    } else if (auto *Do = dyn_cast<DoStmt>(Statement)) {
      push(StatementKind::DoWhile, Statement);
      rc_recur flattenStatement(Do->getBody());
    } else if (auto *For = dyn_cast<ForStmt>(Statement)) {
      push(StatementKind::For, Statement);
      rc_recur flattenStatement(For->getBody());
    } else if (auto *Switch = dyn_cast<SwitchStmt>(Statement)) {
      push(StatementKind::Switch, Statement);
      rc_recur flattenStatement(Switch->getBody());
    } else if (isa<ReturnStmt>(Statement)) {
      push(StatementKind::Return, Statement);
    } else if (isa<DeclStmt>(Statement)) {
      push(StatementKind::LocalVariableDeclaration, Statement);
    } else if (isa<GotoStmt>(Statement)) {
      push(StatementKind::Goto, Statement);
    } else if (isa<BreakStmt>(Statement)) {
      push(StatementKind::Break, Statement);
    } else if (isa<ContinueStmt>(Statement)) {
      push(StatementKind::Continue, Statement);
    } else if (isa<Expr>(Statement)) {
      push(StatementKind::Expression, Statement);
    } else {
      Output.Error = std::string("unsupported C statement in the provided C "
                                 "code: ")
                     + Statement->getStmtClassName();
    }
  }

  void attachComments(clang::ASTContext &Context, clang::FileID MainFile) {
    const auto *Comments = Context.Comments.getCommentsInFile(MainFile);
    if (Comments == nullptr)
      return;

    StringRef Buffer = Sources->getBufferData(MainFile);

    struct RawEntry {
      unsigned Begin = 0;
      unsigned End = 0;
      StringRef Raw;
    };
    std::vector<RawEntry> Entries;
    for (const auto &[Offset, Comment] : *Comments) {
      StringRef Raw = Comment->getRawText(*Sources);
      auto RangeBegin = Comment->getSourceRange().getBegin();
      unsigned Begin = Sources->getFileOffset(RangeBegin);
      Entries.push_back({ Begin, Begin + unsigned(Raw.size()), Raw });
    }
    llvm::sort(Entries, [](const RawEntry &Left, const RawEntry &Right) {
      return Left.Begin < Right.Begin;
    });

    // Both statements (pre-order) and comments are sorted by begin offset, so a
    // single forward cursor over the comments suffices: each statement claims
    // the run of full-line comments directly above it, and any comment we move
    // past without claiming can no longer match anything and is dropped.
    size_t Next = 0;
    for (CStatement &Statement : Output.Statements) {
      // Comments entirely before this statement are the candidates; the rest
      // begin at or after it and belong to a later statement.
      size_t Last = Next;
      while (Last < Entries.size()
             and Entries[Last].End <= Statement.BeginOffset)
        ++Last;

      // Collect the contiguous block of full-line comments immediately above
      // the statement, nearest first. A break (code in the gap, or a comment
      // trailing a statement's line) discards everything above it.
      llvm::SmallVector<StringRef> Collected;
      unsigned Boundary = Statement.BeginOffset;
      for (size_t Index = Last; Index > Next;) {
        --Index;
        const RawEntry &Entry = Entries[Index];
        StringRef Gap = Buffer.substr(Entry.End, Boundary - Entry.End);
        if (not llvm::all_of(Gap, llvm::isSpace))
          break;
        if (not startsOwnLine(Buffer, Entry.Begin))
          break;
        Collected.push_back(Entry.Raw);
        Boundary = Entry.Begin;
      }
      Next = Last;

      llvm::SmallVector<std::string> Lines;
      for (StringRef Raw : llvm::reverse(Collected))
        appendCommentLines(Raw, Lines);
      Statement.LeadingComments = std::move(Lines);
    }
  }
};

/// Capture Clang error diagnostics into a string.
class DiagnosticCollector : public clang::DiagnosticConsumer {
  std::string &Output;

public:
  DiagnosticCollector(std::string &Output) : Output(Output) {}

  void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                        const clang::Diagnostic &Info) override {
    clang::DiagnosticConsumer::HandleDiagnostic(Level, Info);
    if (Level < clang::DiagnosticsEngine::Error)
      return;

    llvm::SmallString<256> Message;
    Info.FormatDiagnostic(Message);
    Output += Message.str().str();
    Output += "\n";
  }
};

class ParseAction : public clang::ASTFrontendAction {
  ParseOutput &Output;
  const model::Binary &Binary;
  llvm::ArrayRef<std::string> RetypeStrings;

public:
  ParseAction(ParseOutput &Output,
              const model::Binary &Binary,
              llvm::ArrayRef<std::string> RetypeStrings) :
    Output(Output), Binary(Binary), RetypeStrings(RetypeStrings) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, StringRef) override {
    return std::make_unique<ParseConsumer>(Output, Binary, RetypeStrings);
  }

  bool BeginInvocation(clang::CompilerInstance &CompilerInstance) override {
    CompilerInstance.getDiagnostics()
      .setClient(new DiagnosticCollector(Output.Error),
                 /*ShouldOwnClient=*/true);
    return true;
  }
};

} // namespace

llvm::Expected<ParsedFunction>
revng::editcbody::parseUserFunction(StringRef HeaderPath,
                                    StringRef CCode,
                                    const model::Binary &Binary) {
  // Prepend a synthetic probe declaration per `RETYPE:` directive, so Clang
  // resolves the types named in the comments in the same parse as the function
  // body.
  std::vector<std::string> RetypeStrings = collectRetypeStrings(CCode);
  std::string Probes;
  for (size_t I = 0; I < RetypeStrings.size(); ++I)
    Probes += makeRetypeProbe(RetypeStrings[I], I);

  std::string Input = ("#include \"" + HeaderPath + "\"\n" + Probes + CCode)
                        .str();
  revng_log(Log, "Parsing:\n" << Input << "\n");

  ParseOutput Output;
  static constexpr StringRef InputFileName = "revng-input.c";
  auto Action = std::make_unique<ParseAction>(Output, Binary, RetypeStrings);
  // edit-c-body needs every comment, not only the documentation ones.
  std::vector<std::string> Flags = revng::getClangCompileFlags();
  Flags.push_back("-fparse-all-comments");
  if (not clang::tooling::runToolOnCodeWithArgs(std::move(Action),
                                                Input,
                                                Flags,
                                                InputFileName))
    return revng::createError("Unable to run clang");

  if (not Output.Error.empty())
    return revng::createError(Output.Error);

  return ParsedFunction{ std::move(Output.Statements),
                         std::move(Output.ResolvedTypes) };
}
