//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "revng/CliftEmitC/CCommentEmitter.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

void mlir::clift::emitHeaderPrologue(ptml::CTokenEmitter &Tokens) {
  Tokens.emitPragmaOnceDirective();
  Tokens.emitNewline();

  Tokens.emitCategoryComment("This header has been generated using rev.ng.");

  // TODO: emit the license information, revng version information, etc.
}

void mlir::clift::emitCommonIncludes(ptml::CTokenEmitter &Tokens) {
  // TODO: attach proper header locations.
  Tokens.emitIncludeDirective("attributes.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitIncludeDirective("primitive-types.h",
                              "",
                              ptml::CTokenEmitter::IncludeMode::Quote);
  Tokens.emitNewline();
}

static void emitTypeGraph(mlir::MLIRContext &Context,
                          const mlir::clift::TypeDependencyGraph &Graph,
                          ptml::CTokenEmitter &Tokens,
                          const TargetCImplementation &Target,
                          mlir::clift::TypeEmitterConfiguration Configuration) {
  mlir::clift::TypeDefinitionEmitter Emitter(Tokens, Target, Configuration);

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

  Tokens.emitNewline();
}

void mlir::clift::emitTypes(ptml::CTokenEmitter &Tokens,
                            const TargetCImplementation &Target,
                            mlir::ModuleOp Module,
                            TypeEmitterConfiguration Configuration) {
  auto Graph = TypeDependencyGraph::makeModelGraph(Module);

  if (not Graph.empty()) {
    Tokens.emitCategoryComment("Types");
    emitTypeGraph(*Module.getContext(), Graph, Tokens, Target, Configuration);
  }
}

template<pipeline::RankSpecialization RankFilter>
void emitFunctionsImpl(ptml::CTokenEmitter &Tokens,
                       const TargetCImplementation &Target,
                       mlir::ModuleOp Module,
                       const RankFilter &Rank,
                       llvm::StringRef CategoryComment) {
  mlir::clift::CEmitter Emitter(Tokens, Target);

  bool CommentEmitted = false;
  Module->walk([&](mlir::clift::FunctionOp Function) {
    if (pipeline::locationFromString(Rank, Function.getHandle())) {
      if (not CommentEmitted) {
        Tokens.emitCategoryComment(CategoryComment);
        CommentEmitted = true;
      }

      Emitter.emitFunctionPrototype(Function);
      Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
      Tokens.emitNewline();
      Tokens.emitNewline();
    }
  });
}

void mlir::clift::emitFunctions(ptml::CTokenEmitter &Tokens,
                                const TargetCImplementation &Target,
                                mlir::ModuleOp Module) {
  emitFunctionsImpl(Tokens,
                    Target,
                    Module,
                    revng::ranks::Function,
                    "Functions");
}

void mlir::clift::emitDynamicFunctions(ptml::CTokenEmitter &Tokens,
                                       const TargetCImplementation &Target,
                                       mlir::ModuleOp Module) {
  emitFunctionsImpl(Tokens,
                    Target,
                    Module,
                    revng::ranks::DynamicFunction,
                    "Imported Dynamic Functions");
}

void mlir::clift::emitSegments(ptml::CTokenEmitter &Tokens,
                               const TargetCImplementation &Target,
                               mlir::ModuleOp Module) {
  CEmitter Emitter(Tokens, Target);

  bool CommentEmitted = false;
  Module->walk([&Tokens,
                &Emitter,
                &CommentEmitted](mlir::clift::GlobalVariableOp Segment) {
    auto MaybeLocation = pipeline::locationFromString(revng::ranks::Segment,
                                                      Segment.getHandle());
    revng_assert(MaybeLocation.has_value());

    if (not CommentEmitted) {
      Tokens.emitCategoryComment("Segments");
      CommentEmitted = true;
    }

    if (auto CommentAttribute = Segment->getAttr("clift.comment")) {
      auto String = mlir::dyn_cast<mlir::StringAttr>(CommentAttribute);
      revng_assert(String != nullptr);

      if (not String.getValue().empty()) {
        CCommentEmitter Comments(Tokens);
        Comments.emitComment(String.getValue());
      }
    }

    static constexpr auto GV = ptml::CTokenEmitter::EntityKind::GlobalVariable;
    Emitter.emitDeclaration(Segment.getType(),
                            CEmitter::DeclaratorInfo{
                              .Identifier = Segment.getName(),
                              .Location = Segment.getHandle(),
                              .Attributes = {},
                              .Kind = GV,
                              .Parameters = {} });

    Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
    Tokens.emitNewline();
    Tokens.emitNewline();
  });
}

void mlir::clift::emitSingleTypeDefinition(mlir::MLIRContext &Context,
                                           ptml::CTokenEmitter &Tokens,
                                           const TargetCImplementation &Target,
                                           mlir::clift::DefinedType Type,
                                           TypeEmitterConfiguration Config) {
  TypeDefinitionEmitter Emitter(Tokens, Target, Config);

  Emitter.emitTypeDefinition(Context, Type);
  Tokens.emitNewline();
}

void mlir::clift::emitHelpers(ptml::CTokenEmitter &Tokens,
                              const TargetCImplementation &Target,
                              std::vector<mlir::ModuleOp> &Modules) {
  // TODO: emit `#include`s

  auto Graph = TypeDependencyGraph::makeHelperGraph(Modules);

  if (not Graph.empty()) {
    Tokens.emitCategoryComment("Types");

    revng_assert(!Modules.empty());
    emitTypeGraph(*Modules.front().getContext(),
                  Graph,
                  Tokens,
                  Target,
                  TypeEmitterConfiguration{
                    .TypeToOmit = {},
                    .PrintMaximumEnumValue = false,
                    .ExplicitPadding = true,
                  });
  }

  CEmitter Emitter(Tokens, Target);

  bool CommentEmitted = false;
  std::unordered_set<std::string_view> EmittedFunctions;
  for (mlir::ModuleOp Module : Modules) {
    Module->walk([&Tokens,
                  &Emitter,
                  &CommentEmitted,
                  &EmittedFunctions](mlir::clift::FunctionOp Function) {
      if (EmittedFunctions.contains(Function.getHandle()))
        return;

      if (pipeline::locationFromString(revng::ranks::HelperFunction,
                                       Function.getHandle())) {
        if (not CommentEmitted) {
          Tokens.emitCategoryComment("Functions");
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
