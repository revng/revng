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

    CEmitter Emitter(Tokens, Target);
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
                     const TypeDependencyGraph &Graph,
                     TypeDefinitionEmitter &Emitter) {
    // In order to improve the printing order, do the visit it in two parts:
    // first only start from nodes without any successors (real roots),
    // only then, resolve potential loops by starting from arbitrary nodes.
    std::unordered_set<const TypeDependencyNode *> Emitted;
    for (const auto *Root : Graph.nodes())
      if (not Root->predecessorCount())
        Emitter.emitTypeTree(Context, *Root, Emitted);
    for (const auto *Root : Graph.nodes())
      Emitter.emitTypeTree(Context, *Root, Emitted);

    revng_assert(Graph.size() == Emitted.size());
  }

public:
  void emitTypes(mlir::ModuleOp Module,
                 TypeEmitterConfiguration Configuration) {
    TypeDefinitionEmitter Emitter(Tokens,
                                  Target,
                                  *Module.getContext(),
                                  Configuration);

    auto Graph = TypeDependencyGraph::makeModelGraph(Module);
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
    CEmitter Emitter(Tokens, Target);

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
    CEmitter Emitter(Tokens, Target);

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
                              CEmitter::DeclaratorInfo{
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

public:
  void emitHelpers(llvm::MutableArrayRef<mlir::ModuleOp> Modules) {
    revng_check(not Modules.empty());

    TypeDefinitionEmitter Emitter(Tokens,
                                  Target,
                                  *Modules.front().getContext(),
                                  TypeEmitterConfiguration{
                                    .TypeToOmit = {},
                                    .EmitMaximumEnumValue = false,
                                    .ExplicitPadding = true,
                                  });

    // TODO: emit `#include`s

    auto Graph = TypeDependencyGraph::makeHelperGraph(Modules);
    if (not Graph.empty()) {
      Emitter.emitCategoryComment("Types");

      revng_assert(!Modules.empty());
      emitTypeGraph(*Modules.front().getContext(), Graph, Emitter);
    }

    bool CommentEmitted = false;
    std::unordered_set<std::string_view> EmittedFunctions;
    for (mlir::ModuleOp Module : Modules) {
      Module->walk([this,
                    &Emitter,
                    &CommentEmitted,
                    &EmittedFunctions](mlir::clift::FunctionOp Function) {
        if (EmittedFunctions.contains(Function.getHandle()))
          return;

        if (pipeline::locationFromString(revng::ranks::HelperFunction,
                                         Function.getHandle())) {
          if (not CommentEmitted) {
            Emitter.emitCategoryComment("Functions");
            CommentEmitted = true;
          }

          Emitter.emitFunctionPrototype(Function);
          Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
          Tokens.emitNewline();
          Tokens.emitNewline();

          auto [_, Success] = EmittedFunctions.emplace(Function.getHandle());
          revng_assert(Success);
        }
      });
    }

    Tokens.emitNewline();
  }
};

void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                             const TargetCImplementation &Target,
                             mlir::ModuleOp Module,
                             TypeEmitterConfiguration Configuration) {
  CHeaderEmitterImpl Emitter(Tokens, Target);

  Emitter.emitHeaderPrologue();
  Emitter.emitCommonIncludes();

  // TODO: split the following into separate headers.
  Emitter.emitTypes(Module, Configuration);
  Emitter.emitFunctions(Module);
  Emitter.emitDynamicFunctions(Module);
  Emitter.emitSegments(Module);
}

void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                      const TargetCImplementation &Target,
                      llvm::MutableArrayRef<mlir::ModuleOp> Modules) {
  CHeaderEmitterImpl Emitter(Tokens, Target);

  Emitter.emitHeaderPrologue();
  Emitter.emitHelpers(Modules);
}
