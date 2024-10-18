//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Segment.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Support/ModelHelpers.h"

#include "../FuncOrCallInst.h"
#include "DLATypeSystemBuilder.h"

using namespace dla;
using namespace llvm;

using TSBuilder = DLATypeSystemLLVMBuilder;

bool TSBuilder::createInterproceduralTypes(llvm::Module &M,
                                           const model::Binary &Model) {
  for (const Function &F : M.functions()) {

    auto FTags = FunctionTags::TagsSet::from(&F);
    // Skip intrinsics
    if (F.isIntrinsic())
      continue;

    // Ignore everything that is not isolated or dynamic
    if (not FunctionTags::Isolated.isTagOf(&F)
        and not FunctionTags::DynamicFunction.isTagOf(&F))
      continue;

    revng_assert(not F.isVarArg());

    // Check if a function with the same prototype has already been visited
    const model::TypeDefinition *Prototype = nullptr;
    if (FunctionTags::Isolated.isTagOf(&F)) {
      const model::Function *ModelFunc = llvmToModelFunction(Model, F);
      Prototype = Model.prototypeOrDefault(ModelFunc->prototype());
    } else {
      llvm::StringRef SymbolName = F.getName().drop_front(strlen("dynamic_"));

      auto It = Model.ImportedDynamicFunctions().find(SymbolName.str());
      revng_assert(It != Model.ImportedDynamicFunctions().end());
      Prototype = Model.prototypeOrDefault(It->prototype());
    }

    revng_assert(Prototype);

    FuncOrCallInst FuncWithSameProto;
    auto It = VisitedPrototypes.find(Prototype);
    if (It == VisitedPrototypes.end())
      VisitedPrototypes.insert({ Prototype, &F });
    else
      FuncWithSameProto = It->second;

    // Create the Function's return types
    auto FRetTypes = getOrCreateLayoutTypes(F);
    // Add equality links between return values of function with the same
    // prototype
    if (not FuncWithSameProto.isNull()) {
      auto OtherRetVals = getLayoutTypes(*FuncWithSameProto.getVal());
      revng_assert(FRetTypes.size() == OtherRetVals.size());
      for (auto [N1, N2] : llvm::zip(OtherRetVals, FRetTypes))
        TS.addEqualityLink(N1, N2.first);
    }

    revng_assert(FuncWithSameProto.isNull()
                 or F.arg_size() == FuncWithSameProto.arg_size());

    const auto *RFT = dyn_cast<model::RawFunctionDefinition>(Prototype);
    const auto *CABIFT = dyn_cast<model::CABIFunctionDefinition>(Prototype);
    if (RFT) {
      revng_assert(F.arg_size() == RFT->Arguments().size()
                   or (not RFT->StackArgumentsType().isEmpty()
                       and (F.arg_size() == RFT->Arguments().size() + 1)));
    } else if (CABIFT) {
      revng_assert(CABIFT->Arguments().size() == F.arg_size());
    } else {
      revng_abort();
    }

    // Create types for the Function's arguments
    for (const auto &ArgVal : F.args()) {
      auto ArgIndex = ArgVal.getArgNo();
      // Arguments can only be integers and pointers
      revng_assert(isa<IntegerType>(ArgVal.getType())
                   or isa<PointerType>(ArgVal.getType()));
      auto [ArgNode, _] = getOrCreateLayoutType(&ArgVal);
      revng_assert(ArgNode);

      model::UpcastableType ArgumentModelType;
      if (RFT) {
        const auto &ModelArgs = RFT->Arguments();
        if (ArgIndex < ModelArgs.size())
          ArgumentModelType = std::next(ModelArgs.begin(), ArgIndex)->Type();
        else
          ArgumentModelType = RFT->StackArgumentsType();
      } else {
        revng_assert(CABIFT);
        ArgumentModelType = CABIFT->Arguments().at(ArgIndex).Type();
      }

      if (model::PointerType *Pointer = ArgumentModelType->getPointer()) {
        if (const model::UpcastableType &Pointee = Pointer->PointeeType()) {
          if (not Pointee->isVoidPrimitive()) {
            bool IsFunction = Pointee->isPrototype();
            bool IsScalar = !IsFunction && Pointee->isScalar();

            auto MaybeSize = Pointee->size();
            bool IsSized = MaybeSize.has_value();

            // If it is not scalar then it must be sized or a function type
            revng_assert(IsFunction or IsScalar or IsSized);

            if (not IsScalar) {
              ArgNode->NonScalar = true;
              if (IsSized)
                ArgNode->Size = *MaybeSize;
              else if (IsFunction)
                ArgNode->Size = getPointerSize(Model.Architecture());
            } else {
              // Skip char, because they alias and propagate weird information.
              if (IsSized and *MaybeSize > 1)
                ArgNode->Size = *MaybeSize;
              else if (IsFunction)
                ArgNode->Size = getPointerSize(Model.Architecture());
            }
          }
        }
      }

      // If there is already a Function with the same prototype, add equality
      // edges between args
      if (not FuncWithSameProto.isNull()) {
        auto &OtherArg = *(FuncWithSameProto.getArg(ArgIndex));
        auto *OtherArgNode = getLayoutType(&OtherArg);
        revng_assert(OtherArgNode);
        TS.addEqualityLink(ArgNode, OtherArgNode);
      }
    }

    for (const BasicBlock &B : F) {
      for (const Instruction &I : B) {
        if (auto *Call = getCallToIsolatedFunction(&I)) {

          const Function *Callee = getCallee(Call);
          if (not Callee)
            continue;

          unsigned ArgNo = 0U;
          for (const Use &ArgUse : Call->args()) {

            // Create the layout for the call arguments
            const auto *ActualArg = ArgUse.get();
            revng_assert(isa<IntegerType>(ActualArg->getType())
                         or isa<PointerType>(ActualArg->getType()));
            auto ActualTypes = getOrCreateLayoutTypes(*ActualArg);

            // Create the layout for the formal arguments.
            Value *FormalArg = Callee->getArg(ArgNo);
            revng_assert(isa<IntegerType>(FormalArg->getType())
                         or isa<PointerType>(FormalArg->getType()));
            auto FormalTypes = getOrCreateLayoutTypes(*FormalArg);
            revng_assert(1ULL == ActualTypes.size() == FormalTypes.size());

            auto FieldNum = FormalTypes.size();
            if (not isa<ConstantInt>(ActualArg)) {
              for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
                TS.addInstanceLink(ActualTypes[FieldId].first,
                                   FormalTypes[FieldId].first,
                                   OffsetExpression{});
                auto *Placeholder = TS.createArtificialLayoutType();
                Placeholder->Size = getPointerSize(Model.Architecture());
                TS.addPointerLink(Placeholder, ActualTypes[FieldId].first);
              }
            }
            ++ArgNo;
          }
        } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
          revng_assert(isa<IntegerType>(PHI->getType())
                       or isa<PointerType>(PHI->getType())
                       or isa<StructType>(PHI->getType()));
          auto PHITypes = getOrCreateLayoutTypes(*PHI);
          for (const Use &Incoming : PHI->incoming_values()) {
            revng_assert(isa<IntegerType>(Incoming->getType())
                         or isa<PointerType>(Incoming->getType())
                         or isa<StructType>(Incoming->getType()));
            auto InTypes = getOrCreateLayoutTypes(*Incoming.get());
            revng_assert(PHITypes.size() == InTypes.size());
            revng_assert((PHITypes.size() == 1ULL)
                         or isa<StructType>(PHI->getType()));
            auto FieldNum = PHITypes.size();
            if (not isa<ConstantInt>(Incoming)) {
              for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
                TS.addInstanceLink(InTypes[FieldId].first,
                                   PHITypes[FieldId].first,
                                   OffsetExpression{});
              }
            }
          }
        } else if (auto *RetI = dyn_cast<ReturnInst>(&I)) {
          if (Value *RetVal = RetI->getReturnValue()) {
            revng_assert(isa<StructType>(RetVal->getType())
                         or isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            auto RetTypes = getOrCreateLayoutTypes(*RetVal);
            revng_assert(RetTypes.size() == FRetTypes.size());
            auto FieldNum = RetTypes.size();
            if (not isa<ConstantInt>(RetVal)) {
              for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
                if (RetTypes[FieldId].first != nullptr) {
                  TS.addInstanceLink(RetTypes[FieldId].first,
                                     FRetTypes[FieldId].first,
                                     OffsetExpression{});
                  auto *Placeholder = TS.createArtificialLayoutType();
                  Placeholder->Size = getPointerSize(Model.Architecture());
                  TS.addPointerLink(Placeholder, RetTypes[FieldId].first);
                }
              }
            }
          }
        }
      }
    }
  }

  // Create types for segments

  const auto &Segments = Model.Segments();

  std::map<const model::Segment *, LayoutTypeSystemNode *> SegmentNodeMap;

  for (const model::Segment &S : Segments) {
    // Initialize a node for every segment
    LayoutTypeSystemNode
      *SegmentNode = SegmentNodeMap[&S] = TS.createArtificialLayoutType();
    // Set the Size, which is known for segments.
    SegmentNode->Size = S.VirtualSize();
    // Set NonScalar to true, so that it cannot be removed from
    // the optimization steps of DLA's middle-end
    SegmentNode->NonScalar = true;
  }

  for (Function &F : FunctionTags::SegmentRef.functions(&M)) {
    const auto &[StartAddress, VirtualSize] = extractSegmentKeyFromMetadata(F);
    const model::Segment *Segment = &Segments.at({ StartAddress, VirtualSize });
    LayoutTypeSystemNode *SegmentNode = SegmentNodeMap.at(Segment);

    LayoutTypeSystemNode *SegmentRefNode = getOrCreateLayoutType(&F).first;

    // The type of the segment and the type returned by segmentref are the same
    TS.addEqualityLink(SegmentNode, SegmentRefNode);

    for (const Use &U : F.uses()) {
      auto *Call = cast<CallInst>(U.getUser());
      LayoutTypeSystemNode *SegmentRefCallNode = getOrCreateLayoutType(Call)
                                                   .first;
      // The type of the segment is also the same as the type of all the calls
      // to the SegmentRef function.
      TS.addEqualityLink(SegmentNode, SegmentRefCallNode);
    }
  }

  for (Function &F : FunctionTags::StringLiteral.functions(&M)) {
    const auto &[StartAddress,
                 VirtualSize,
                 Offset,
                 StrLen,
                 _] = extractStringLiteralFromMetadata(F);

    const model::Segment *Segment = &Segments.at({ StartAddress, VirtualSize });
    LayoutTypeSystemNode *SegmentNode = SegmentNodeMap.at(Segment);

    LayoutTypeSystemNode *LiteralNode = getOrCreateLayoutType(&F).first;

    // We have an instance of the literal at Offset inside the type of the
    // segment itself.
    TS.addInstanceLink(SegmentNode, LiteralNode, dla::OffsetExpression(Offset));

    LayoutTypeSystemNode *ByteType = TS.createArtificialLayoutType();
    ByteType->Size = 1;
    dla::OffsetExpression OE{};
    OE.Offset = 0;
    OE.Strides.push_back(ByteType->Size);
    OE.TripCounts.push_back(1 + StrLen);
    // The type of the literal contains, as offset zero a stride of Strlen+1
    // instances of ByteType.
    TS.addInstanceLink(LiteralNode, ByteType, std::move(OE));

    for (const Use &U : F.uses()) {
      auto *Call = cast<CallInst>(U.getUser());
      LayoutTypeSystemNode *StringLiteralCall = getOrCreateLayoutType(Call)
                                                  .first;
      // The type of each call to the StringLiteral function has an instance of
      // a ByteType at the beginning.
      // This roughly translates the idea that the call to StringLiteral returns
      // a char * in C.
      TS.addInstanceLink(StringLiteralCall, ByteType, OffsetExpression{ 0 });
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());
  return TS.getNumLayouts() != 0;
}
