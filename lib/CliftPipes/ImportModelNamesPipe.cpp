//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Helpers.h"
#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftPipes/CliftContainer.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Support/Identifier.h"

namespace clift = mlir::clift;
namespace rr = revng::ranks;

namespace {

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
      AttrLists.emplace_back(Op.getArgAttrs(I));
    }
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

public:
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
class NameImporter : public clift::ModuleVisitor<NameImporter> {
  struct CurrentFunctionState {
    using LocationType = pipeline::Location<decltype(rr::Function)>;

    LocationType Location;
    model::CNameBuilder::VariableNameBuilder Variables;
    model::CNameBuilder::GotoLabelNameBuilder GotoLabels;

    explicit CurrentFunctionState(NameImporter &Importer,
                                  LocationType &&Location,
                                  const model::Function &Function) :
      Location(std::move(Location)),
      Variables(Importer.NameBuilder.localVariables(Function)),
      GotoLabels(Importer.NameBuilder.gotoLabels(Function)) {}
  };

  const model::Binary &Model;
  SymbolRenamer &Symbols;
  model::CNameBuilder NameBuilder;

  std::optional<CurrentFunctionState> CurrentFunction;

public:
  explicit NameImporter(const model::Binary &Model, SymbolRenamer &Symbols) :
    Model(Model), Symbols(Symbols), NameBuilder(Model) {}

  //===---------------------- ModuleVisitor interface ---------------------===//

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (auto T = mlir::dyn_cast<clift::FunctionType>(Type)) {
      if (pipeline::locationFromString(rr::HelperFunction, T.getHandle())) {
        // Do nothing
      } else {
        const model::TypeDefinition *MT = getModelType(T.getHandle(),
                                                       rr::TypeDefinition);
        revng_assert(MT != nullptr);

        T.getMutableName().setValue(sanitizeIdentifier(NameBuilder.name(*MT)));
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

    revng_abort("Unsupported type location");
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (auto S = mlir::dyn_cast<clift::MakeLabelOp>(Op))
      return visitMakeLabelOp(S);

    if (auto S = mlir::dyn_cast<clift::LocalVariableOp>(Op))
      return visitLocalVariableOp(S);

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
    }

    return mlir::success();
  }

  mlir::LogicalResult visitTypeDefinition(clift::TypeDefinitionAttr T,
                                          const model::TypeDefinition &MT) {
    T.getMutableName().setValue(NameBuilder.name(MT));

    if (auto ST = mlir::dyn_cast<clift::StructAttr>(T))
      return importStructNames(ST, llvm::cast<model::StructDefinition>(MT));

    if (auto UT = mlir::dyn_cast<clift::UnionAttr>(T)) {
      const auto &UMT = llvm::cast<model::UnionDefinition>(MT);

      for (auto [I, F] : llvm::enumerate(UT.getFields())) {
        const auto &Field = UMT.Fields().at(static_cast<uint64_t>(I));
        F.getMutableName().setValue(NameBuilder.name(UMT, Field));
      }

      return mlir::success();
    }

    if (auto ET = mlir::dyn_cast<clift::EnumAttr>(T)) {
      const auto &EMT = llvm::cast<model::EnumDefinition>(MT);

      for (auto E : ET.getFields()) {
        const auto &Entry = EMT.Entries().at(E.getRawValue());
        E.getMutableName().setValue(NameBuilder.name(EMT, Entry));
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

    std::string
      Name = (Model.Configuration().Naming().ArtificialReturnValuePrefix()
              + NameBuilder.name(FMT));

    ST.getMutableName().setValue(Name);

    for (auto [F, R] : llvm::zip(ST.getFields(), FMT.ReturnValues()))
      F.getMutableName().setValue(NameBuilder.name(FMT, R));

    return mlir::success();
  }

  mlir::LogicalResult
  visitRawStackArguments(clift::StructAttr ST,
                         const model::RawFunctionDefinition &FMT) {
    const auto &SAT = *FMT.StackArgumentsType();
    const auto *SMT = getModelType<model::StructDefinition>(SAT);
    revng_assert(SMT != nullptr);

    ST.getMutableName().setValue(NameBuilder.name(FMT));

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
    auto AddressSet = getUserAddressSet(Op);

    auto R = CurrentFunction->GotoLabels.name(AddressSet);
    auto L = CurrentFunction->Location.extend(rr::GotoLabel, R.Index);

    setStringAttr(Op, "clift.handle", L.toString());
    setStringAttr(Op, "clift.name", sanitizeIdentifier(R.Name));

    return mlir::success();
  }

  mlir::LogicalResult visitLocalVariableOp(clift::LocalVariableOp Op) {
    auto AddressSet = getUserAddressSet(Op);

    auto R = CurrentFunction->Variables.name(AddressSet);
    auto L = CurrentFunction->Location.extend(rr::LocalVariable, R.Index);

    setStringAttr(Op, "clift.handle", L.toString());
    setStringAttr(Op, "clift.name", sanitizeIdentifier(R.Name));

    return mlir::success();
  }

  mlir::LogicalResult visitFunctionOp(clift::FunctionOp Op) {
    CurrentFunction.reset();

    if (auto Pair = getModelFunction(Op.getHandle())) {
      auto &[L, MF] = *Pair;

      const model::TypeDefinition *Type = getModelType(*MF.Prototype());

      CurrentFunction.emplace(*this, std::move(L), MF);
      Symbols.record(Op, NameBuilder.name(MF));

      ArgumentAttributeMutator Attrs(Op);

      using CF = model::CABIFunctionDefinition;
      using RF = model::RawFunctionDefinition;

      if (const auto *T = llvm::dyn_cast<CF>(Type)) {
        revng_assert(Op.getArgCount() == T->Arguments().size());
        auto TL = pipeline::location(rr::TypeDefinition, T->key());

        for (auto [I, A] : llvm::enumerate(T->Arguments())) {
          auto AL = TL.extend(rr::CABIArgument, static_cast<uint64_t>(I));
          Attrs.setString(I, "clift.handle", AL.toString());
          Attrs.setString(I, "clift.name", NameBuilder.name(*T, A));
        }
      } else if (const auto *T = llvm::dyn_cast<RF>(Type)) {
        bool HasStackArgument = static_cast<bool>(T->StackArgumentsType());

        size_t ArgumentCount = T->Arguments().size() + HasStackArgument;
        revng_assert(Op.getArgCount() == ArgumentCount);

        auto TL = pipeline::location(rr::TypeDefinition, T->key());
        for (auto [I, A] : llvm::enumerate(T->Arguments())) {
          auto AL = TL.extend(rr::RawArgument, A.Location());
          Attrs.setString(I, "clift.handle", AL.toString());
          Attrs.setString(I, "clift.name", NameBuilder.name(*T, A));
        }

        if (HasStackArgument) {
          unsigned I = T->Arguments().size();

          auto AL = TL.transmute(rr::RawStackArguments);
          auto Name = Model.Configuration().Naming().RawStackArgumentName();

          Attrs.setString(I, "clift.handle", AL.toString());
          Attrs.setString(I, "clift.name", Name);
        }
      } else {
        revng_abort("Invalid function prototype");
      }

      Attrs.commit();

      return mlir::success();
    }

    if (auto Pair = getModelDynamicFunction(Op.getHandle())) {
      Symbols.record(Op, NameBuilder.name(Pair->Object));
      return mlir::success();
    }

    if (auto L = pipeline::locationFromString(rr::HelperFunction,
                                              Op.getHandle())) {
      Symbols.record(Op, sanitizeIdentifier(L->back()));
      return mlir::success();
    }

    revng_abort("Invalid function handle");
  }

  mlir::LogicalResult visitGlobalVariableOp(clift::GlobalVariableOp Op) {
    if (const model::Segment *Segment = getModelSegment(Op)) {
      Symbols.record(Op, NameBuilder.name(Model, *Segment));
      return mlir::success();
    }

    revng_abort("Invalid global variable handle");
  }
};

class ImportModelNamesPipe {
public:
  static constexpr auto Name = "import-model-names";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(CliftFunction,
                                      0,
                                      CliftFunction,
                                      0,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           revng::pipes::CliftContainer &CliftContainer) {
    mlir::ModuleOp Module = CliftContainer.getModule();
    const model::Binary &Model = *revng::getModelFromContext(EC);

    std::unordered_map<MetaAddress, clift::FunctionOp> Functions;
    Module->walk([&Functions](clift::FunctionOp F) {
      MetaAddress MA = getMetaAddress(F);
      if (MA.isValid()) {
        auto [Iterator, Inserted] = Functions.try_emplace(MA, F);
        revng_assert(Inserted);
      }
    });

    SymbolRenamer Symbols;
    for (const model::Function &Function :
         revng::getFunctionsAndCommit(EC, CliftContainer.name())) {
      auto It = Functions.find(Function.Entry());
      revng_check(It != Functions.end(), "Requested Clift function not found");

      auto R = NameImporter::visit(It->second, Model, Symbols);
      revng_assert(R.succeeded());
    }

    for (mlir::Operation &Op : Module.getBody()->getOperations()) {
      if (auto F = mlir::dyn_cast<clift::FunctionOp>(Op)) {
        if (getMetaAddress(F).isInvalid()) {
          auto R = NameImporter::visit(F, Model, Symbols);
          revng_assert(R.succeeded());
        }
      } else if (auto G = mlir::dyn_cast<clift::GlobalVariableOp>(Op)) {
        auto R = NameImporter::visit(G, Model, Symbols);
        revng_assert(R.succeeded());
      }
    }

    Symbols.apply(Module);
  }
};

static pipeline::RegisterPipe<ImportModelNamesPipe> X;

} // namespace
