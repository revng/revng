//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>
#include <type_traits>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftTypes.h"
#include "revng/Clift/Helpers.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftImportModel/CAttributeListBuilder.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/PTML/CommentPlacementHelper.h"
#include "revng/Support/Identifier.h"

namespace rr = revng::ranks;

namespace {

struct CliftStatementTraits {
  using StatementType = mlir::Operation *;

  static auto getStatements(mlir::Block *Block) {
    using Iterator = mlir::Block::OpListType::iterator;
    using IteratorRange = llvm::iterator_range<Iterator>;
    return llvm::map_range(IteratorRange(Block->getOperations()),
                           [](mlir::Operation &Op) { return &Op; });
  }

  static auto getAddresses(mlir::Operation *Op) {
    std::set<MetaAddress> AddressSet;

    auto GatherRegionAddresses = [&AddressSet](mlir::Region &Region) {
      Region.walk([&AddressSet](mlir::Operation *Op) {
        revng_assert(mlir::isa<clift::ExpressionOpInterface>(Op));
        if (auto Loc = mlir::dyn_cast_or_null<mlir::NameLoc>(Op->getLoc())) {
          if (auto L = pipeline::locationFromString(rr::Instruction,
                                                    Loc.getName().str())) {
            revng_assert(L->back().isValid());
            AddressSet.insert(L->back());
          }
        }
      });
    };

    if (auto ERI = mlir::dyn_cast<clift::ExpressionRegionOpInterface>(Op)) {
      for (mlir::Region &Region : ERI.getExpressionRegions())
        GatherRegionAddresses(Region);
    }
    return AddressSet;
  }
};

struct CliftStatementTreeTraits {
  using TreeType = mlir::Block *;
  using TreeNodeType = mlir::Block *;

  static mlir::Block *getTree(mlir::Block *Block) { return Block; }
  static mlir::Block *getTreeRoot(mlir::Block *Block) { return Block; }
  static mlir::Block *getNode(mlir::Block *Block) { return Block; }
};

// Helper class for mutating the attribute dictionary of a function parameter.
// All attributes associated with a given function parameter are stored in a
// dictionary attribute, which is by its nature immutable. Changing individual
// function parameter attributes is difficult and inefficient. This class allows
// changes to all function parameter attribute dictionaries to be aggregated and
// applied all at once.
class ArgumentAttributeMutator {
  clift::FunctionOp Function;
  llvm::SmallVector<mlir::NamedAttrList> AttrLists;

public:
  explicit ArgumentAttributeMutator(clift::FunctionOp Op) : Function(Op) {
    for (unsigned I = 0; I < Op.getArgCount(); ++I) {
      AttrLists.emplace_back(Op.getArgAttrs(I).getDictionaryAttr());
    }
  }

  mlir::Attribute get(unsigned Index, llvm::StringRef Name) const {
    return AttrLists[Index].get(Name);
  }

  void set(unsigned Index, llvm::StringRef Name, mlir::Attribute Attr) {
    AttrLists[Index].set(Name, Attr);
  }

  void setString(unsigned Index, llvm::StringRef Name, llvm::StringRef Value) {
    set(Index, Name, mlir::StringAttr::get(Function.getContext(), Value));
  }

  void commit() {
    llvm::SmallVector<mlir::Attribute> ArgAttrs;
    for (const mlir::NamedAttrList &AttrList : AttrLists)
      ArgAttrs.push_back(AttrList.getDictionary(Function.getContext()));

    Function.setArgAttrsAttr(mlir::ArrayAttr::get(Function.getContext(),
                                                  ArgAttrs));
  }
};

// Helper class used for recording symbol renames and applying them all at once.
class SymbolRenamer {
  llvm::DenseMap<llvm::StringRef, std::string> Map;

  // Set of module-level ops whose descriptive info has already been
  // recorded. Lives here (and not on the Importer) because a single
  // importDescriptiveInfo invocation may construct several Importer
  // instances (e.g. the per-function overload visits the current
  // function, then each helper/global, then any callee reached through a
  // clift::UseOp); the dedup gate must be shared across all of them so
  // that the same target is not recorded twice.
  llvm::DenseSet<mlir::Operation *> RecordedTargets;

public:
  // Mark `Op` as recorded; return true if this is the first time it is
  // seen during the current importDescriptiveInfo call, false otherwise.
  // Callers that perform per-op work (prototype attributes, comments,
  // ...) should use this to gate that work so it only happens once.
  bool markRecorded(clift::GlobalOpInterface Op) {
    return RecordedTargets.insert(Op.getOperation()).second;
  }

  void record(clift::GlobalOpInterface Op, llvm::StringRef NewName) {
    auto [Iterator, Inserted] = Map.try_emplace(Op.getName(), NewName.str());
    revng_assert(Inserted);
  }

  void apply(mlir::ModuleOp Module) {
    Module->walk([this](mlir::Operation *Op) {
      if (auto Global = mlir::dyn_cast<clift::GlobalOpInterface>(Op)) {
        if (auto It = Map.find(Global.getName()); It != Map.end()) {
          Global.setName(It->second);
        }
      } else if (auto Use = mlir::dyn_cast<clift::UseOp>(Op)) {
        if (auto It = Map.find(Use.getSymbolName()); It != Map.end()) {
          Use.setSymbolName(It->second);
        }
      }
    });

    Map.clear();
  }
};

// Visitor used for applying names to operations, types and their members found
// by visiting a given operation and all nested operations. Any module-level
// operations are not renamed directly, but instead the renames are recorded
// in the specified SymbolRenamer, to be applied all at once.
class Importer : public clift::ModuleVisitor<Importer> {
  struct CurrentFunctionState {
    using LocationType = pipeline::Location<decltype(rr::Function)>;

    const model::Function &Model;
    LocationType Location;
    model::CNameBuilder::VariableNameBuilder Variables;
    model::CNameBuilder::GotoLabelNameBuilder GotoLabels;

    yield::CommentPlacementHelper<mlir::Block *,
                                  CliftStatementTreeTraits,
                                  CliftStatementTraits>
      Comments;

    explicit CurrentFunctionState(Importer &Importer,
                                  clift::FunctionOp Function,
                                  LocationType &&Location,
                                  const model::Function &ModelFunction) :
      Model(ModelFunction),
      Location(std::move(Location)),
      Variables(Importer.NameBuilder.localVariables(ModelFunction)),
      GotoLabels(Importer.NameBuilder.gotoLabels(ModelFunction)),
      Comments(ModelFunction, &Function.getBody().front()) {}
  };

  const model::Binary &Model;
  SymbolRenamer &Symbols;
  model::CNameBuilder NameBuilder;

  std::optional<CurrentFunctionState> CurrentFunction;

public:
  explicit Importer(const model::Binary &Model, SymbolRenamer &Symbols) :
    Model(Model), Symbols(Symbols), NameBuilder(Model) {}

  //===---------------------- ModuleVisitor interface ---------------------===//

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (auto T = mlir::dyn_cast<clift::FunctionType>(Type)) {
      if (auto L = pipeline::locationFromString(rr::HelperFunction,
                                                T.getHandle())) {
        T.getMutableName().setValue(sanitizeIdentifier(L->back()));

      } else {
        const model::TypeDefinition *MT = getModelType(T.getHandle(),
                                                       rr::TypeDefinition);
        revng_assert(MT != nullptr,
                     ("Unknown type: '" + T.getHandle().str() + "'").c_str());

        T.getMutableName().setValue(NameBuilder.name(*MT));
        T.getMutableComment().setValue(MT->Comment());
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult visitAttr(mlir::Attribute Attr) {
    auto T = mlir::dyn_cast<clift::TypeDefinitionAttr>(Attr);
    if (not T)
      return mlir::success();

    if (const auto *MT = getModelType(T.getHandle(), rr::TypeDefinition))
      return visitTypeDefinition(T, *MT);

    if (const auto *MT = getModelType(T.getHandle(), rr::ArtificialStruct)) {
      const auto *FMT = llvm::cast<model::RawFunctionDefinition>(MT);
      return visitArtificialStruct(mlir::cast<clift::StructAttr>(T), *FMT);
    }

    if (const auto *MT = getModelType(T.getHandle(), rr::RawStackArguments)) {
      const auto *FMT = llvm::cast<model::RawFunctionDefinition>(MT);
      return visitRawStackArguments(mlir::cast<clift::StructAttr>(T), *FMT);
    }

    if (auto L = pipeline::locationFromString(rr::HelperStructType,
                                              T.getHandle())) {
      return visitHelperStructType(mlir::cast<clift::StructAttr>(Attr), *L);
    }

    if (auto L = pipeline::locationFromString(rr::OpaqueType, T.getHandle())) {
      return visitOpaqueType(mlir::cast<clift::StructAttr>(Attr), *L);
    }

    revng_abort("Unsupported type location");
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    revng_assert(CurrentFunction.has_value());
    if (auto S = mlir::dyn_cast<clift::MakeLabelOp>(Op))
      return visitMakeLabelOp(S);

    if (auto S = mlir::dyn_cast<clift::LocalVariableOp>(Op))
      return visitLocalVariableOp(S);

    if (auto S = mlir::dyn_cast<clift::StatementOpInterface>(Op))
      return visitStatementOp(S);

    if (auto U = mlir::dyn_cast<clift::UseOp>(Op))
      return visitUseOp(U);

    return mlir::success();
  }

  // Resolve the symbol referenced by `Use` to its definition in the
  // enclosing module and record the descriptive info for that target.
  //
  // This is what guarantees that a function whose body contains
  // references to other functions or to global variables also causes
  // those callees to be renamed by the SymbolRenamer at the end of
  // importDescriptiveInfo, without the per-function variant having to
  // pre-walk every sibling module-level op.
  mlir::LogicalResult visitUseOp(clift::UseOp Use) {
    auto Module = Use->getParentOfType<mlir::ModuleOp>();
    revng_assert(Module);

    mlir::Operation *Target = //
      mlir::SymbolTable::lookupSymbolIn(Module, Use.getSymbolNameAttr());

    if (auto F = mlir::dyn_cast_or_null<clift::FunctionOp>(Target))
      return recordFunctionOpName(F);

    if (auto G = mlir::dyn_cast_or_null<clift::GlobalVariableOp>(Target))
      return recordGlobalVariableOpName(G);

    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op))
      return visitFunctionOp(F);

    if (auto G = mlir::dyn_cast<clift::GlobalVariableOp>(Op))
      return visitGlobalVariableOp(G);

    return mlir::success();
  }

private:
  //===----------------------------- Utilities ----------------------------===//

  template<typename RankT, typename ObjectT>
  struct LocationObjectPair {
    pipeline::Location<RankT> Location;
    const ObjectT &Object;
  };

  template<typename RankT, typename ContainerT>
  std::optional<LocationObjectPair<RankT, typename ContainerT::value_type>>
  getModelObject(llvm::StringRef Handle,
                 const RankT &Rank,
                 const ContainerT &Container) {
    using PairType = LocationObjectPair<RankT, typename ContainerT::value_type>;

    if (auto L = pipeline::locationFromString(Rank, Handle)) {
      const auto &[Key] = L->back();
      auto It = Container.find(Key);
      if (It != Container.end())
        return std::optional<PairType>(std::in_place, *L, *It);
    }
    return std::nullopt;
  }

  template<typename RankT>
  const model::TypeDefinition *
  getModelType(llvm::StringRef Handle, const RankT &Rank) {
    if (auto L = pipeline::locationFromString(Rank, Handle)) {
      auto It = Model.TypeDefinitions().find(L->back());
      if (It != Model.TypeDefinitions().end())
        return It->get();
    }
    return nullptr;
  }

  template<typename TypeDefinitionT = model::TypeDefinition>
  const TypeDefinitionT *getModelType(const model::Type &Type) {
    if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type))
      return llvm::dyn_cast<TypeDefinitionT>(D->Definition().get());
    return nullptr;
  }

  //===------------------------- Type name import -------------------------===//

  mlir::LogicalResult importStructNames(clift::StructAttr ST,
                                        const model::StructDefinition &SMT) {
    for (auto F : ST.getFields()) {
      const auto &Field = SMT.Fields().at(F.getOffset());
      F.getMutableName().setValue(NameBuilder.name(SMT, Field));
      F.getMutableComment().setValue(Field.Comment());
    }

    return mlir::success();
  }

  mlir::LogicalResult visitTypeDefinition(clift::TypeDefinitionAttr T,
                                          const model::TypeDefinition &MT) {
    T.getMutableName().setValue(NameBuilder.name(MT));
    T.getMutableComment().setValue(MT.Comment());

    if (auto ST = mlir::dyn_cast<clift::StructAttr>(T))
      return importStructNames(ST, llvm::cast<model::StructDefinition>(MT));

    if (auto UT = mlir::dyn_cast<clift::UnionAttr>(T)) {
      const auto &UMT = llvm::cast<model::UnionDefinition>(MT);

      for (auto [I, F] : llvm::enumerate(UT.getFields())) {
        const auto &Field = UMT.Fields().at(static_cast<uint64_t>(I));
        F.getMutableName().setValue(NameBuilder.name(UMT, Field));
        F.getMutableComment().setValue(Field.Comment());
      }

      return mlir::success();
    }

    if (auto ET = mlir::dyn_cast<clift::EnumAttr>(T)) {
      const auto &EMT = llvm::cast<model::EnumDefinition>(MT);

      for (auto E : ET.getFields()) {
        const auto &Entry = EMT.Entries().at(E.getRawValue());
        E.getMutableName().setValue(NameBuilder.name(EMT, Entry));
        E.getMutableComment().setValue(Entry.Comment());
      }

      return mlir::success();
    }

    if (auto TT = mlir::dyn_cast<clift::TypedefAttr>(T))
      return mlir::success();

    revng_abort("Unsupported type definition attribute.");
  }

  mlir::LogicalResult
  visitArtificialStruct(clift::StructAttr ST,
                        const model::RawFunctionDefinition &FMT) {
    revng_assert(ST.getFields().size() == FMT.ReturnValues().size());

    std::string Name = NameBuilder.artificialReturnValueWrapperName(FMT);

    ST.getMutableName().setValue(Name);

    for (auto [F, R] : llvm::zip(ST.getFields(), FMT.ReturnValues())) {
      F.getMutableName().setValue(NameBuilder.name(FMT, R));
      F.getMutableComment().setValue(R.Comment());
    }

    return mlir::success();
  }

  mlir::LogicalResult
  visitRawStackArguments(clift::StructAttr ST,
                         const model::RawFunctionDefinition &FMT) {
    const auto &SAT = *FMT.StackArgumentsType();
    const auto *SMT = getModelType<model::StructDefinition>(SAT);
    revng_assert(SMT != nullptr);

    ST.getMutableName().setValue(NameBuilder.name(FMT));
    ST.getMutableComment().setValue(SMT->Comment());

    return importStructNames(ST, *SMT);
  }

  mlir::LogicalResult
  visitHelperStructType(clift::StructAttr ST,
                        const pipeline::Location<decltype(rr::HelperStructType)>
                          &L) {
    std::string
      Name = (Model.Configuration().Naming().ArtificialReturnValuePrefix()
              + sanitizeIdentifier(L.back()));

    ST.getMutableName().setValue(Name);

    for (auto [I, F] : llvm::enumerate(ST.getFields())) {
      std::string Name;
      {
        llvm::raw_string_ostream Out(Name);
        Out << "field_" << I;
      }
      F.getMutableName().setValue(Name);
    }

    return mlir::success();
  }

  mlir::LogicalResult
  visitOpaqueType(clift::StructAttr ST,
                  const pipeline::Location<decltype(rr::OpaqueType)> &L) {
    ST.getMutableName().setValue(NameBuilder.opaqueTypeName(ST.getSize()));
    return mlir::success();
  }

  //===----------------------- Operation name import ----------------------===//

  auto getModelFunction(llvm::StringRef Handle) {
    return getModelObject(Handle, rr::Function, Model.Functions());
  }

  auto getModelDynamicFunction(llvm::StringRef Handle) {
    return getModelObject(Handle,
                          rr::DynamicFunction,
                          Model.ImportedDynamicFunctions());
  }

  const model::Segment *getModelSegment(clift::GlobalVariableOp Op) {
    auto L = pipeline::locationFromString(rr::Segment, Op.getHandle());
    if (not L)
      return nullptr;

    auto Key = L->at(rr::Segment);
    auto It = Model.Segments().find(Key);
    if (It == Model.Segments().end())
      return nullptr;

    return &*It;
  }

  static void setStringAttr(mlir::Operation *Op,
                            llvm::StringRef Name,
                            llvm::StringRef Value) {
    Op->setAttr(Name, mlir::StringAttr::get(Op->getContext(), Value));
  }

  SortedVector<MetaAddress> getUserAddressSet(mlir::Value Value) {
    auto GetMetaAddress = [](mlir::Operation *Op) {
      if (auto Loc = mlir::dyn_cast_or_null<mlir::NameLoc>(Op->getLoc())) {
        if (auto L = pipeline::locationFromString(rr::Instruction,
                                                  Loc.getName().str())) {
          revng_assert(L->back().isValid());
          return L->back();
        }
      }
      return MetaAddress::invalid();
    };

    SortedVector<MetaAddress> AddressSet;
    for (const auto &User : Value.getUsers()) {
      MetaAddress Address = GetMetaAddress(User);

      if (not Address.isValid()) {
        AddressSet.clear();
        break;
      }

      AddressSet.emplace(Address);
    }
    return AddressSet;
  }

  mlir::LogicalResult visitMakeLabelOp(clift::MakeLabelOp Op) {
    if (auto L = pipeline::locationFromString(rr::GotoLabel, Op.getHandle())) {
      Op.setName(CurrentFunction->GotoLabels.name(getUserAddressSet(Op)).Name);
    } else {
      Op.setName(CurrentFunction->GotoLabels.automaticName().Name);
    }

    // TODO: label comments.

    return mlir::success();
  }

  mlir::LogicalResult visitLocalVariableOp(clift::LocalVariableOp Op) {
    if (auto L = pipeline::locationFromString(rr::StackFrameVariable,
                                              Op.getHandle())) {
      Op.setName(NameBuilder.name(CurrentFunction->Model.StackFrame()));
    } else if (auto L = pipeline::locationFromString(rr::LocalVariable,
                                                     Op.getHandle())) {
      Op.setName(CurrentFunction->Variables.name(getUserAddressSet(Op)).Name);
    } else {
      Op.setName(CurrentFunction->Variables.automaticName().Name);
    }

    // TODO: variable comments.

    return mlir::success();
  }

private:
  void visitFunctionPrototypeImpl(auto Location,
                                  clift::FunctionOp Op,
                                  const model::TypeDefinition *Prototype,
                                  llvm::StringRef Name,
                                  llvm::StringRef Comment) {
    ArgumentAttributeMutator Attrs(Op);

    using CF = model::CABIFunctionDefinition;
    using RF = model::RawFunctionDefinition;

    llvm::StringRef ReturnValueComment = "";
    if (const auto *T = llvm::dyn_cast<CF>(Prototype)) {
      revng_assert(Op.getArgCount() == T->Arguments().size());
      auto TL = pipeline::location(rr::TypeDefinition, T->key());

      for (auto [I, A] : llvm::enumerate(T->Arguments())) {
        auto AL = TL.extend(rr::CABIArgument, static_cast<uint64_t>(I));
        Attrs.setString(I, "clift.handle", AL.toString());
        Attrs.setString(I, "clift.name", NameBuilder.name(*T, A));
        Attrs.setString(I, "clift.comment", A.Comment());
      }

      ReturnValueComment = T->ReturnValueComment();

    } else if (const auto *T = llvm::dyn_cast<RF>(Prototype)) {
      bool HasStackArgument = static_cast<bool>(T->StackArgumentsType());

      size_t ArgumentCount = T->Arguments().size() + HasStackArgument;
      revng_assert(Op.getArgCount() == ArgumentCount);

      auto TL = pipeline::location(rr::TypeDefinition, T->key());
      for (auto [I, A] : llvm::enumerate(T->Arguments())) {
        auto AL = TL.extend(rr::RawArgument, A.Location());
        Attrs.setString(I, "clift.handle", AL.toString());
        Attrs.setString(I, "clift.name", NameBuilder.name(*T, A));
        Attrs.setString(I, "clift.comment", A.Comment());

        std::string RegisterName = toString(A.Location());

        // TODO: consider using a dedicated
        //       `/register/$architecture/$name` location.
        auto RegisterLocation = "";

        clift::CAttributeListBuilder Attributes{
          Op.getContext(),
          Attrs.get(I, "clift.c_attributes"),
        };
        Attributes.setOrUpdate<"_REG">(RegisterName, RegisterLocation);
        Attrs.set(I, "clift.c_attributes", Attributes.get());
      }

      if (HasStackArgument) {
        unsigned I = T->Arguments().size();

        auto AL = TL.transmute(rr::RawStackArguments);
        auto Name = Model.Configuration().Naming().RawStackArgumentName();

        Attrs.setString(I, "clift.handle", AL.toString());
        Attrs.setString(I, "clift.name", Name);

        clift::CAttributeListBuilder Attributes{
          Op.getContext(),
          Attrs.get(I, "clift.c_attributes"),
        };
        Attributes.setOrUpdate<"_STACK">();
        Attrs.set(I, "clift.c_attributes", Attributes.get());
      }

      ReturnValueComment = T->ReturnValueComment();

    } else {
      revng_abort("Invalid function prototype");
    }

    Attrs.commit();

    Symbols.record(Op, Name);
    Op->setAttr("clift.comment",
                mlir::StringAttr::get(Op->getContext(), Comment));
    Op->setAttr("clift.return_value_comment",
                mlir::StringAttr::get(Op->getContext(), ReturnValueComment));
  }

public:
  // Record the descriptive info for the given function op, without touching
  // the `CurrentFunction` state used by the body walk. This is the part of
  // the work that can be performed for any function op the importer comes
  // across (the current function, a sibling declaration, a callee reached
  // through a `clift::UseOp`, ...).
  //
  // Idempotent: a second call on the same op is a no-op.
  mlir::LogicalResult recordFunctionOpName(clift::FunctionOp Op) {
    if (not Symbols.markRecorded(Op))
      return mlir::success();

    if (auto Pair = getModelFunction(Op.getHandle())) {
      auto &[L, MF] = *Pair;
      const auto *Prototype = Model.prototypeOrDefault(MF.prototype());
      visitFunctionPrototypeImpl(L,
                                 Op,
                                 Prototype,
                                 NameBuilder.name(MF),
                                 MF.Comment());

      return mlir::success();
    }

    if (auto Pair = getModelDynamicFunction(Op.getHandle())) {
      auto &[L, MF] = *Pair;
      const auto *Prototype = Model.prototypeOrDefault(MF.prototype());
      visitFunctionPrototypeImpl(L,
                                 Op,
                                 Prototype,
                                 NameBuilder.name(MF),
                                 MF.Comment());

      return mlir::success();
    }

    if (auto L = pipeline::locationFromString(rr::HelperFunction,
                                              Op.getHandle())) {
      Symbols.record(Op, sanitizeIdentifier(L->back()));
      return mlir::success();
    }

    revng_abort("Invalid function handle");
  }

  mlir::LogicalResult visitFunctionOp(clift::FunctionOp Op) {
    CurrentFunction.reset();

    if (recordFunctionOpName(Op).failed())
      return mlir::failure();

    if (auto Pair = getModelFunction(Op.getHandle())) {
      auto &[L, MF] = *Pair;
      if (not Op.getBody().empty())
        CurrentFunction.emplace(*this, Op, std::move(L), MF);
    }

    return mlir::success();
  }

  // Record the descriptive info for the given global variable op. Mirrors
  // recordFunctionOpName: contains no per-function state and is therefore
  // safe to call from any visit context (module level or while walking a
  // function body following a clift::UseOp).
  //
  // Idempotent: a second call on the same op is a no-op.
  mlir::LogicalResult recordGlobalVariableOpName(clift::GlobalVariableOp Op) {
    if (not Symbols.markRecorded(Op))
      return mlir::success();

    if (const model::Segment *Segment = getModelSegment(Op)) {
      Symbols.record(Op, NameBuilder.name(Model, *Segment));

      // No need to use symbol renamer for comments, as they don't affect any
      // users.
      Op->setAttr("clift.comment",
                  mlir::StringAttr::get(Op->getContext(), Segment->Comment()));

      return mlir::success();
    }

    revng_abort("Invalid global variable handle");
  }

  mlir::LogicalResult visitGlobalVariableOp(clift::GlobalVariableOp Op) {
    return recordGlobalVariableOpName(Op);
  }

  //===-------------------------- Comment import --------------------------===//

  mlir::LogicalResult visitStatementOp(clift::StatementOpInterface Op) {
    const auto &Comments = CurrentFunction->Comments.getComments(Op);

    if (not Comments.empty()) {
      llvm::SmallVector<mlir::Attribute> CommentAttrList;

      const auto &ModelComments = CurrentFunction->Model.Comments();
      for (const auto &Comment : Comments) {
        auto Body = ModelComments.at(Comment.CommentIndex).Body();
        CommentAttrList.push_back(mlir::StringAttr::get(Op->getContext(),
                                                        Body));
      }

      Op->setAttr("clift.comments",
                  mlir::ArrayAttr::get(Op->getContext(), CommentAttrList));
    }

    return mlir::success();
  }
};

} // namespace

void clift::importDescriptiveInfo(const model::Binary &Model,
                                  mlir::ModuleOp Module) {
  SymbolRenamer Symbols;

  auto R = Importer::visit(Module, Model, Symbols);
  revng_assert(R.succeeded());

  Symbols.apply(Module);
}

void clift::importDescriptiveInfo(const model::Function &Function,
                                  const model::Binary &Model,
                                  mlir::ModuleOp Module) {
  std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
  clift::FunctionOp CliftFunction = nullptr;
  Module->walk([&Function, &CliftFunction](clift::FunctionOp F) {
    MetaAddress MA = getMetaAddress(F);
    if (Function.Entry() == MA) {
      revng_check(CliftFunction == nullptr);
      CliftFunction = F;
    }
  });
  revng_check(CliftFunction != nullptr, "Requested Clift function not found");

  SymbolRenamer Symbols;

  auto R = Importer::visit(CliftFunction, Model, Symbols);
  revng_assert(R.succeeded());

  for (mlir::Operation &Op : Module.getBody()->getOperations()) {
    if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op)) {
      if (getMetaAddress(F).isInvalid()) {
        auto R = Importer::visit(F, Model, Symbols);
        revng_assert(R.succeeded());
      }
    } else if (auto G = mlir::dyn_cast<clift::GlobalVariableOp>(Op)) {
      auto R = Importer::visit(G, Model, Symbols);
      revng_assert(R.succeeded());
    }
  }

  Symbols.apply(Module);
}
