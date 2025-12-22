//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

class CHeaderEmitterImpl {
private:
  ptml::CTokenEmitter &Tokens;
  const TargetCImplementation &Target;

public:
  CHeaderEmitterImpl(ptml::CTokenEmitter &Tokens,
                     const TargetCImplementation &Target) :
    Tokens(Tokens), Target(Target) {}

public:
  void emitHeaderPrologue() {
    // TODO: emit header location *definition*, so that ctrl+click on includes
    //       (references) leads to the correct file.

    Tokens.emitPragmaOnceDirective();
    Tokens.emitNewline();

    mlir::clift::CEmitter Emitter(Tokens, Target);
    Emitter.emitCategoryComment("This header has been generated using rev.ng.");

    // TODO: emit the license information, revng version information, etc.
  }

  void emitCommonIncludes() {
    // TODO: attach proper header *reference* locations.

    Tokens.emitIncludeDirective("attributes.h",
                                "",
                                ptml::CTokenEmitter::IncludeMode::Quote);
    Tokens.emitIncludeDirective("primitive-types.h",
                                "",
                                ptml::CTokenEmitter::IncludeMode::Quote);
    Tokens.emitNewline();
  }

private:
  void emitTypeGraph(mlir::MLIRContext &Context,
                     const mlir::clift::TypeDependencyGraph &Graph,
                     mlir::clift::TypeDefinitionEmitter &Emitter) {
    // In order to improve the printing order, do the visit it in two parts:
    // first only start from nodes without any successors (real roots),
    // only then, resolve potential loops by starting from arbitrary nodes.
    std::unordered_set<const mlir::clift::TypeDependencyNode *> Emitted;
    for (const auto *Root : Graph.nodes())
      if (not Root->predecessorCount())
        Emitter.emitTypeTree(Context, *Root, Emitted);
    for (const auto *Root : Graph.nodes())
      Emitter.emitTypeTree(Context, *Root, Emitted);

    revng_assert(Graph.size() == Emitted.size());
  }

public:
  void emitTypes(mlir::ModuleOp Module,
                 mlir::clift::TypeEmitterConfiguration Configuration) {
    mlir::clift::TypeDefinitionEmitter Emitter(Tokens,
                                               Target,
                                               *Module.getContext(),
                                               Configuration);

    auto Graph = mlir::clift::TypeDependencyGraph::makeModelGraph(Module);
    if (not Graph.empty()) {
      Emitter.emitCategoryComment("Types");
      emitTypeGraph(*Module.getContext(), Graph, Emitter);
      Tokens.emitNewline();
    }
  }

private:
  template<pipeline::RankSpecialization RankFilter>
  void emitFunctionsImpl(mlir::ModuleOp Module,
                         const RankFilter &Rank,
                         llvm::StringRef CategoryComment) {
    mlir::clift::CEmitter Emitter(Tokens, Target);

    bool CommentEmitted = false;
    Module->walk([&](mlir::clift::FunctionOp Function) {
      if (pipeline::locationFromString(Rank, Function.getHandle())) {
        if (not CommentEmitted) {
          Emitter.emitCategoryComment(CategoryComment);
          CommentEmitted = true;
        }

        Emitter.emitFunctionPrototype(Function);
        Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
        Tokens.emitNewline();
        Tokens.emitNewline();
      }
    });
  }

public:
  void emitFunctions(mlir::ModuleOp Module) {
    emitFunctionsImpl(Module, revng::ranks::Function, "Functions");
  }

  void emitDynamicFunctions(mlir::ModuleOp Module) {
    emitFunctionsImpl(Module,
                      revng::ranks::DynamicFunction,
                      "Imported Dynamic Functions");
  }

public:
  void emitSegments(mlir::ModuleOp Module) {
    mlir::clift::CEmitter Emitter(Tokens, Target);

    bool CommentEmitted = false;
    Module->walk([this,
                  &Emitter,
                  &CommentEmitted](mlir::clift::GlobalVariableOp Segment) {
      auto MaybeLocation = pipeline::locationFromString(revng::ranks::Segment,
                                                        Segment.getHandle());
      revng_assert(MaybeLocation.has_value());

      if (not CommentEmitted) {
        Emitter.emitCategoryComment("Segments");
        CommentEmitted = true;
      }

      Emitter.emitGlobalDoxygenComment(Segment);

      static constexpr auto G = ptml::CTokenEmitter::EntityKind::GlobalVariable;
      Emitter.emitDeclaration(Segment.getType(),
                              mlir::clift::CEmitter::DeclaratorInfo{
                                .Identifier = Segment.getName(),
                                .Location = Segment.getHandle(),
                                .CAttributes = {},
                                .Kind = G,
                                .Parameters = {} });

      Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      Tokens.emitNewline();
      Tokens.emitNewline();
    });
  }
};

void mlir::clift::emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                                          const TargetCImplementation &Target,
                                          mlir::ModuleOp Module,
                                          TypeEmitterConfiguration
                                            Configuration) {
  CHeaderEmitterImpl Emitter(Tokens, Target);

  Emitter.emitHeaderPrologue();
  Emitter.emitCommonIncludes();

  // TODO: split the following into separate headers.
  Emitter.emitTypes(Module, Configuration);
  Emitter.emitFunctions(Module);
  Emitter.emitDynamicFunctions(Module);
  Emitter.emitSegments(Module);
}
