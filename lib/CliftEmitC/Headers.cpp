//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/Headers.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/CliftEmitC/TypeDependencyGraph.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

class OpaqueTypeSizeCollector
  : public clift::ModuleVisitor<OpaqueTypeSizeCollector> {
private:
  using Base = clift::ModuleVisitor<OpaqueTypeSizeCollector>;

private:
  std::set<uint64_t> &Result;

public:
  OpaqueTypeSizeCollector(std::set<uint64_t> &Result) : Result(Result) {
    // TODO: do not hard-code this list here. Instead, share it with the model
    //       verification logic (the only other current user of this).
    Result = { 1, 2, 4, 8, 10, 12, 16 };
  }

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (uint64_t Size = clift::getObjectSizeOrZero(Type))
      Result.emplace(Size);

    return mlir::success();
  }
};

class CHeaderEmitterImpl : TypeDefinitionEmitter {

public:
  explicit CHeaderEmitterImpl(ptml::CTokenEmitter &Tokens,
                              const CDataModel &DataModel,
                              TypeEmitterConfiguration Configuration) :
    TypeDefinitionEmitter(Tokens, DataModel, Configuration) {}

public:
  void emitHeaderPrologue() {
    // TODO: emit header location *definition*, so that ctrl+click on includes
    //       (references) leads to the correct file.

    Tokens.emitPragmaOnceDirective();
    Tokens.emitNewline();

    emitCategoryComment("This header has been generated using rev.ng.");

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
  void emitTypeGraph(const TypeDependencyGraph &Graph) {
    // In order to improve the printing order, do the visit it in two parts:
    // first only start from nodes without any successors (real roots),
    // only then, resolve potential loops by starting from arbitrary nodes.
    std::unordered_set<const TypeDependencyNode *> Emitted;
    for (const auto *Root : Graph.nodes())
      if (not Root->predecessorCount())
        emitTypeTree(*Root, Emitted);
    for (const auto *Root : Graph.nodes())
      emitTypeTree(*Root, Emitted);

    revng_assert(Graph.size() == Emitted.size());
  }

public:
  void emitTypes(mlir::ModuleOp Module) {
    auto Graph = TypeDependencyGraph::makeModelGraph(Module);
    if (not Graph.empty()) {
      emitCategoryComment("Types");
      emitTypeGraph(Graph);
      Tokens.emitNewline();
    }
  }

private:
  template<pipeline::RankSpecialization RankFilter>
  void emitFunctionsImpl(mlir::ModuleOp Module,
                         const RankFilter &Rank,
                         llvm::StringRef CategoryComment) {
    bool CommentEmitted = false;
    Module->walk([&](clift::FunctionOp Function) {
      if (pipeline::locationFromString(Rank, Function.getHandle())) {
        if (not CommentEmitted) {
          emitCategoryComment(CategoryComment);
          CommentEmitted = true;
        }

        emitFunctionPrototype(Function);
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
    bool CommentEmitted = false;
    Module->walk([this, &CommentEmitted](clift::GlobalVariableOp Segment) {
      auto MaybeLocation = pipeline::locationFromString(revng::ranks::Segment,
                                                        Segment.getHandle());
      revng_assert(MaybeLocation.has_value());

      if (not CommentEmitted) {
        emitCategoryComment("Segments");
        CommentEmitted = true;
      }

      {
        using RegionKind = ptml::CTokenEmitter::RegionKind;
        auto Guard = Tokens.enterRegion(RegionKind::Commentable,
                                        Segment.getHandle());

        emitGlobalDoxygenComment(Segment);

        using EntityKind = ptml::CTokenEmitter::EntityKind;
        emitDeclaration(Segment.getType(),
                        CEmitter::DeclaratorInfo{
                          .Identifier = Segment.getName(),
                          .Location = Segment.getHandle(),
                          .CAttributes = {},
                          .Kind = EntityKind::GlobalVariable,
                          .Parameters = {} });

        Tokens.emitPunctuator(ptml::CTokenEmitter::Punctuator::Semicolon);
        Tokens.emitNewline();
      }

      Tokens.emitNewline();
    });
  }

  template<RangeOf<uint64_t> ByteSizeList = std::initializer_list<uint64_t>>
  void emitOpaqueTypes(mlir::MLIRContext &Context, const ByteSizeList &Sizes) {
    bool CommentEmitted = false;
    for (uint64_t Size : Sizes) {
      if (not CommentEmitted) {
        emitCategoryComment("Opaque types");
        CommentEmitted = true;
      }

      auto Struct = clift::makeOpaqueStruct(Context, Size);
      emitClassDefinition(Struct);
      emitTypeDeclaration(Struct);
    }

    Tokens.emitNewline();
  }

  std::set<uint64_t>
  collectOpaqueByteSizes(llvm::ArrayRef<mlir::ModuleOp> Modules) {
    std::set<uint64_t> Result;

    for (mlir::ModuleOp Module : Modules) {
      auto R = OpaqueTypeSizeCollector::visit(Module, Result);
      revng_assert(R.succeeded());
    }

    return Result;
  }

public:
  void emitHelpers(llvm::ArrayRef<mlir::ModuleOp> Modules) {
    // TODO: emit `#include`s

    auto Graph = TypeDependencyGraph::makeHelperGraph(Modules);
    if (not Graph.empty()) {
      emitCategoryComment("Types");

      revng_assert(!Modules.empty());
      emitTypeGraph(Graph);
    }

    bool CommentEmitted = false;
    std::unordered_set<std::string_view> EmittedFunctions;
    for (mlir::ModuleOp Module : Modules) {
      Module->walk([this,
                    &CommentEmitted,
                    &EmittedFunctions](clift::FunctionOp Function) {
        if (EmittedFunctions.contains(Function.getHandle()))
          return;

        if (pipeline::locationFromString(revng::ranks::HelperFunction,
                                         Function.getHandle())) {
          if (not CommentEmitted) {
            emitCategoryComment("Functions");
            CommentEmitted = true;
          }

          emitFunctionPrototype(Function);
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

void emitCommonIncludes(ptml::CTokenEmitter &Tokens,
                        const CDataModel &DataModel) {
  CHeaderEmitterImpl Emitter(Tokens, DataModel, TypeEmitterConfiguration{});
  Emitter.emitCommonIncludes();
}

void emitTypes(ptml::CTokenEmitter &Tokens,
               mlir::ModuleOp Module,
               TypeEmitterConfiguration Configuration) {
  CHeaderEmitterImpl Emitter(Tokens,
                             clift::getDataModel(Module),
                             Configuration);
  Emitter.emitTypes(Module);
}

void emitTypeAndGlobalHeader(ptml::CTokenEmitter &Tokens,
                             mlir::ModuleOp Module,
                             TypeEmitterConfiguration Configuration,
                             bool DefineOpaqueTypes) {
  CHeaderEmitterImpl Emitter(Tokens,
                             clift::getDataModel(Module),
                             Configuration);

  Emitter.emitHeaderPrologue();
  Emitter.emitCommonIncludes();

  // TODO: split the following into separate headers.
  Emitter.emitTypes(Module);
  Emitter.emitFunctions(Module);
  Emitter.emitDynamicFunctions(Module);
  Emitter.emitSegments(Module);

  if (DefineOpaqueTypes)
    Emitter.emitOpaqueTypes(*Module.getContext(),
                            Emitter.collectOpaqueByteSizes({ Module }));
}

void emitHelperHeader(ptml::CTokenEmitter &Tokens,
                      llvm::ArrayRef<mlir::ModuleOp> Modules) {
  revng_check(not Modules.empty());

  const CDataModel &DataModel = clift::getDataModel(Modules.front());
  revng_assert(llvm::all_of(Modules.drop_front(),
                            [&DataModel](mlir::ModuleOp Module) {
                              return clift::getDataModel(Module) == DataModel;
                            }));

  TypeEmitterConfiguration Configuration = {
    .ExplicitPadding = true,
  };

  CHeaderEmitterImpl Emitter(Tokens, DataModel, Configuration);

  Emitter.emitHeaderPrologue();
  Emitter.emitHelpers(Modules);

  // This trick is practically a `const_cast`.
  mlir::MLIRContext &Context = *mlir::ModuleOp(Modules.front()).getContext();

  Emitter.emitOpaqueTypes(Context, Emitter.collectOpaqueByteSizes(Modules));
}

void emitSingleTypeDefinition(ptml::CTokenEmitter &Tokens,
                              const CDataModel &DataModel,
                              clift::DefinedType Type,
                              TypeEmitterConfiguration Config) {
  TypeDefinitionEmitter Emitter(Tokens, DataModel, Config);

  Emitter.emitTypeDefinition(Type);
  Tokens.emitNewline();
}
