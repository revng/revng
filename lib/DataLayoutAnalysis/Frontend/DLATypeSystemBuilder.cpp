
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"

#include "DLATypeSystemBuilder.h"

using namespace llvm;
using namespace dla;

// We use \l here instead of \n, because graphviz has this sick way of saying
// that the text in the node labels should be left-justified
static constexpr const char DoRet[] = "\\l";

void LLVMTSDebugPrinter::printNodeContent(const LayoutTypeSystem &TS,
                                          const LayoutTypeSystemNode *N,
                                          raw_fd_ostream &File) const {
  auto EqClasses = TS.getEqClasses();
  revng_assert(not EqClasses.isRemoved(N->ID));

  File << DoRet;

  auto HasAssociatedVal = [this](unsigned ID) { return (ID < Values.size()); };

  const auto &CollapsedNodes = EqClasses.computeEqClass(N->ID);
  File << "Collapsed Nodes (# " << CollapsedNodes.size() << ") :" << DoRet;

  for (auto ID : CollapsedNodes) {
    File << "{ ID: " << ID << ", ";
    if (HasAssociatedVal(ID)) {
      const LayoutTypePtr &Val = Values[ID];
      File << "Associated Value: ";

      if (Val.isEmpty())
        File << "Empty (Access Node)";
      else
        Val.print(File);

    } else {
      File << " Artificial";
    }
    File << " }" << DoRet;
  }
}

static void assertGetLayoutTypePreConditions(const Value &V) {
  const Type *VTy = V.getType();
  // We accept only the following types:
  // * pointers, because they are obviously pointers; the associated layout is
  //   the pointee of such pointers
  // * integers, because they may be used to do integer-based pointer arithmetic
  //   which makes them pointers in practice; the associated layout is the
  //   pointee of such pointers
  // * struct types and function types, because struct types are used to
  //   represent return types of Raw functions returning sets of registers, one
  //   register for each field of the struct; so the index tells what field we
  //   are considering; the associated layout is the pointee of such field
  // * array of bytes, because they are used as return types of CABI functions
  //   returning big aggregates;
  //   TODO: these are accepted but not handled properly at this time. proper
  //   handling would require them to be treated roughly like struct types, but
  //   it is considerably harder since we don't have the same strong guarantees
  //   we have with register sets that each field is a register, since here we
  //   can have any nested aggregate type.
  revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy)
               or isa<StructType>(VTy) or isa<FunctionType>(VTy)
               or isArrayOfBytes(VTy));
}

static void assertGetLayoutTypePreConditions(const Value *V, unsigned Id) {
  assertGetLayoutTypePreConditions(*V);
  // The only case where we accept Id != max are Functions that return
  // structs
  revng_assert(Id == std::numeric_limits<unsigned>::max()
               or cast<Function>(V)->getReturnType()->isStructTy());
}

LayoutTypeSystemNode *DLATypeSystemLLVMBuilder::getLayoutType(const Value *V,
                                                              unsigned Id) {

  if (V == nullptr)
    return nullptr;

  // Check pre-conditions
  assertGetLayoutTypePreConditions(V, Id);

  LayoutTypePtr Key(V, Id);
  return VisitedValues.at(Key);
}

std::pair<LayoutTypeSystemNode *, bool>
DLATypeSystemLLVMBuilder::getOrCreateLayoutType(const Value *V, unsigned Id) {

  if (V == nullptr)
    return std::make_pair(nullptr, false);

  // Check pre-conditions
  assertGetLayoutTypePreConditions(V, Id);

  LayoutTypePtr Key(V, Id);
  auto HintIt = VisitedValues.lower_bound(Key);
  if (HintIt != VisitedValues.end()
      and not VisitedValues.key_comp()(Key, HintIt->first)) {
    return std::make_pair(HintIt->second, false);
  }

  LayoutTypeSystemNode *Res = TS.createArtificialLayoutType();

  VisitedValues.emplace_hint(HintIt, Key, Res);
  return std::make_pair(Res, true);
}

SmallVector<LayoutTypeSystemNode *, 2>
DLATypeSystemLLVMBuilder::getLayoutTypes(const Value &V) {
  assertGetLayoutTypePreConditions(V);

  SmallVector<LayoutTypeSystemNode *, 2> Results;

  const Type *VTy = V.getType();
  if (const auto *F = dyn_cast<Function>(&V)) {
    auto *RetTy = F->getReturnType();
    if (auto *StructTy = dyn_cast<StructType>(RetTy)) {
      unsigned FieldId = 0;
      unsigned FieldNum = StructTy->getNumElements();
      for (; FieldId < FieldNum; ++FieldId) {
        auto FieldTy = StructTy->getElementType(FieldId);
        revng_assert(isa<IntegerType>(FieldTy) or isa<PointerType>(FieldTy));
        Results.push_back(getLayoutType(&V, FieldId));
      }
    } else if (isArrayOfBytes(RetTy)) {
      // TODO: A function returning an array-of-bytes is a CABI function
      // returning an aggregate. In order to handle it properly we should
      // visit it recursively and have first-class support for model types
      // in DLA, which we currently don't have.
      // So, for now we just bail out, accepting to degrade the quality of
      // the results. We don't expect this to affect users much in
      // practice, because either the input has no debug symbols (and in
      // that case DLA sees mostly Raw functions, and this case doesn't
      // trigger) or the model (via debug symbols, or user input) has lots
      // of good information on CABI function types, in which case the
      // role of DLA is not that important interprocedurally.
      revng_assert(Results.empty());
    } else {
      revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy));
      Results.push_back(getLayoutType(&V));
    }
  } else if (auto *StructTy = dyn_cast<StructType>(VTy)) {
    revng_assert(not isa<LoadInst>(V));

    if (isa<CallInst>(&V) or isa<PHINode>(&V)) {

      // Special handling for StructInitializers
      const Function *Callee = getCallee(cast<Instruction>(&V));
      if (Callee and FunctionTags::StructInitializer.isTagOf(Callee)) {
        revng_assert(not Callee->isVarArg());

        auto *RetTy = cast<StructType>(Callee->getReturnType());
        revng_assert(RetTy->getNumElements() == Callee->arg_size());

        bool OnlyReturnUses = true;
        bool HasReturnUse = false;
        auto *Call = cast<CallInst>(&V);
        for (const User *U : Call->users()) {
          if (isa<ReturnInst>(U)) {
            HasReturnUse = true;

            const Function *Caller = Call->getFunction();

            if (Results.empty())
              Results = getLayoutTypes(*Caller);
            else
              revng_assert(Results == getLayoutTypes(*Caller));

            revng_assert(Results.size() == Callee->arg_size());
          } else {
            OnlyReturnUses = false;
          }
        }
        revng_assert(not HasReturnUse or OnlyReturnUses);
      }

      // If Results are full, we have detected a call to a struct_initializer
      // that is returned, so we are done. Otherwise the have to look to for
      // extractvalue instructions that are extracting values from the return
      // value of the struct_initializer call.
      if (Results.empty()) {

        auto *I = cast<Instruction>(&V);
        const auto ExtractedValues = getExtractedValuesFromInstruction(I);

        Results.resize(ExtractedValues.size(), {});

        for (auto &Group : enumerate(ExtractedValues)) {
          const auto &ExtractedSet = Group.value();
          const auto FieldId = Group.index();
          // Inside here we're working on a single field of the struct.
          // ExtractedSet contains all the ExtractValueInst that extract the
          // same field of the struct. getOrCreateLayoutTypes keeps them as
          // distinct nodes connected by equality links (later merged by
          // CollapseEqualitySCC), so any of them is an equally-good
          // representative for the field: we just take the first one.
          LayoutTypeSystemNode *FieldNode = nullptr;
          for (const CallInst *Ext : ExtractedSet) {
            revng_assert(isCallToTagged(Ext, FunctionTags::OpaqueExtractValue));
            if (FieldNode == nullptr)
              FieldNode = getLayoutType(Ext);
          }
          Results[FieldId] = FieldNode;
        }
      }

    } else {
      Results.resize(StructTy->getNumElements(), nullptr);
    }
  } else {
    // For non-struct and non-function types we only add a
    // LayoutTypeSystemNode
    Results.push_back(getLayoutType(&V));
  }
  return Results;
}

SmallVector<std::pair<LayoutTypeSystemNode *, bool>, 2>
DLATypeSystemLLVMBuilder::getOrCreateLayoutTypes(const Value &V) {
  assertGetLayoutTypePreConditions(V);
  using GetOrCreateResult = std::pair<LayoutTypeSystemNode *, bool>;

  SmallVector<GetOrCreateResult, 2> Results;

  const Type *VTy = V.getType();
  if (const auto *F = dyn_cast<Function>(&V)) {
    auto *RetTy = F->getReturnType();
    if (auto *StructTy = dyn_cast<StructType>(RetTy)) {
      unsigned FieldId = 0;
      unsigned FieldNum = StructTy->getNumElements();
      for (; FieldId < FieldNum; ++FieldId) {
        auto FieldTy = StructTy->getElementType(FieldId);
        revng_assert(isa<IntegerType>(FieldTy) or isa<PointerType>(FieldTy));
        Results.push_back(getOrCreateLayoutType(&V, FieldId));
      }
    } else if (isArrayOfBytes(RetTy)) {
      // TODO: A function returning an array-of-bytes is a CABI function
      // returning an aggregate. In order to handle it properly we should
      // visit it recursively and have first-class support for model types
      // in DLA, which we currently don't have.
      // So, for now we just bail out, accepting to degrade the quality of
      // the results. We don't expect this to affect users much in
      // practice, because either the input has no debug symbols (and in
      // that case DLA sees mostly Raw functions, and this case doesn't
      // trigger) or the model (via debug symbols, or user input) has lots
      // of good information on CABI function types, in which case the
      // role of DLA is not that important interprocedurally.
    } else {
      revng_assert(isa<IntegerType>(VTy) or isa<PointerType>(VTy));
      Results.push_back(getOrCreateLayoutType(&V));
    }
  } else if (auto *StructTy = dyn_cast<StructType>(VTy)) {
    revng_assert(not isa<LoadInst>(V));

    if (isa<CallInst>(&V) or isa<PHINode>(&V)) {

      // Special handling for StructInitializers
      const Function *Callee = getCallee(cast<Instruction>(&V));
      if (Callee and FunctionTags::StructInitializer.isTagOf(Callee)) {
        revng_assert(not Callee->isVarArg());

        auto *RetTy = cast<StructType>(Callee->getReturnType());
        revng_assert(RetTy->getNumElements() == Callee->arg_size());

        bool OnlyReturnUses = true;
        bool HasReturnUse = false;
        auto *Call = cast<CallInst>(&V);
        for (const User *U : Call->users()) {
          if (isa<ReturnInst>(U)) {
            HasReturnUse = true;

            const Function *Caller = Call->getFunction();

            if (Results.empty())
              Results = getOrCreateLayoutTypes(*Caller);
            else
              revng_assert(Results == getOrCreateLayoutTypes(*Caller));

            revng_assert(Results.size() == Callee->arg_size());
          } else {
            OnlyReturnUses = false;
          }
        }
        revng_assert(not HasReturnUse or OnlyReturnUses);
      }

      // If Results are full, we have detected a call to a struct_initializer
      // that is returned, so we are done. Otherwise the have to look to for
      // extractvalue instructions that are extracting values from the return
      // value of the struct_initializer call.
      if (Results.empty()) {

        auto *I = cast<Instruction>(&V);
        const auto ExtractedValues = getExtractedValuesFromInstruction(I);

        Results.resize(ExtractedValues.size(), {});

        for (auto &Group : enumerate(ExtractedValues)) {
          const auto &ExtractedSet = Group.value();
          const auto FieldId = Group.index();
          // Inside here we're working on a single field of the struct.
          // ExtractedSet contains all the ExtractValueInst that extract the
          // same field of the struct.
          // We get or create a layout type for each of them, but they should
          // all be the same.
          std::optional<GetOrCreateResult> FieldResult;
          for (const CallInst *Ext : ExtractedSet) {
            revng_assert(isCallToTagged(Ext, FunctionTags::OpaqueExtractValue));
            GetOrCreateResult ExtResult = getOrCreateLayoutType(Ext);
            if (FieldResult.has_value()) {
              auto &[Node, New] = FieldResult.value();
              const auto &[ExtNode, ExtNew] = ExtResult;
              revng_assert(not ExtNew or ExtNode);
              if (not Node) {
                Node = ExtNode;
              } else if (ExtNode and ExtNode != Node) {
                bool AddedLink = TS.addEqualityLink(Node, ExtNode).second;
                New |= AddedLink;
              }
              New |= ExtNew;
            } else {
              FieldResult = ExtResult;
            }
          }
          Results[FieldId] = FieldResult.value_or(GetOrCreateResult{});
        }
      }

    } else {
      Results.resize(StructTy->getNumElements(), { nullptr, false });
    }
  } else {
    // For non-struct and non-function types we only add a
    // LayoutTypeSystemNode
    Results.push_back(getOrCreateLayoutType(&V));
  }
  return Results;
}

void DLATypeSystemLLVMBuilder::createValuesList() {
  // TODO: the fact that AccessNodes are now added by the frontend means that
  // after initialization not all nodes in the graph correspond to a Value.
  // Can we prevent this?
  this->Values.resize(TS.getNID());

  for (auto &MapIt : VisitedValues) {
    LayoutTypePtr Ptr = MapIt.first;
    unsigned NodeID = MapIt.second->ID;

    revng_assert(NodeID < Values.size());
    this->Values[NodeID] = Ptr;
  }
}

void DLATypeSystemLLVMBuilder::dumpValuesMapping(const StringRef Name) const {
  std::error_code EC;
  raw_fd_ostream OutFile(Name, EC);
  {
    using namespace std::string_literals;
    revng_check(not EC, ("Cannot open: "s + Name.str()).c_str());
  }

  OutFile << "ID; Value; EqClass\n";

  for (auto *N : TS.getLayoutsRange()) {
    // Print Node's ID
    OutFile << N->ID << ";";

    // Check if it has an associated LayoutTypePtr
    if (N->ID < Values.size()) {
      auto &V = Values[N->ID];

      if (V.isEmpty())
        OutFile << "Empty (Access Node)";
      else
        V.print(OutFile);
    } else {
      OutFile << "Out of bounds (No associated LayoutTypePtr)";
    }

    OutFile << ";";

    // Print ID of the node's equivalence class
    if (TS.getEqClasses().getNumClasses() == 0) {
      // Uncompressed
      OutFile << TS.getEqClasses().findLeader(N->ID);
    } else {
      // Compressed
      auto Class = TS.getEqClasses().getEqClassID(N->ID);

      if (Class)
        OutFile << *Class;
      else
        OutFile << "Removed";
    }

    OutFile << "\n";
  }
}
