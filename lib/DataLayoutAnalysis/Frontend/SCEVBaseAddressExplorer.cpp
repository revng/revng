//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Instruction.h"

#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"

#include "SCEVBaseAddressExplorer.h"

static bool isConstantAddress(const llvm::ConstantInt *C) {
  // TODO: insert some logic to properly detect when this constant represents
  // a pointer pointing into some segment
  return false;
}

// Returns true if the Value \p V is always an address, false otherwise.
//
// For now the only Values that are always considered as addresses are calls to
// `revng_stack_frame`, that returns a pointer to the stack of the Function
// that contains the call. A pointer to the stack is obviously always an
// address.
static bool isAlwaysAddress(const llvm::Value *V) {
  if (auto *Call = dyn_cast_or_null<llvm::CallInst>(V))
    if (auto *Callee = Call->getCalledFunction())
      if (FunctionTags::MallocLike.isTagOf(Callee)
          or FunctionTags::AddressOf.isTagOf(Callee))
        return true;
  return false;
}

std::set<const llvm::SCEV *>
SCEVBaseAddressExplorer::findBases(llvm::ScalarEvolution *SE,
                                   const llvm::SCEV *Root,
                                   const SCEVTypeMap &M) {
  std::set<const llvm::SCEV *> Result;
  Worklist.clear();
  Worklist.push_back(Root);

  while (not Worklist.empty()) {
    const llvm::SCEV *AddressCandidate = Worklist.pop_back_val();
    const llvm::SCEV *AddrSCEV = nullptr;
    if (const auto *C = dyn_cast<llvm::SCEVConstant>(AddressCandidate)) {
      // Constants are considered addresses only in case they point to some
      // segment. They are never traversed.
      auto *ConstantVal = C->getValue();
      if (isConstantAddress(ConstantVal))
        AddrSCEV = AddressCandidate;
    } else {
      auto NTraversed = checkAddressOrTraverse(SE, AddressCandidate);
      if (NTraversed) {
        // If we have traversed AddressCandidate, it means that it doesn't
        // look like an address SCEV, so we want to keep looking in is
        // operands to find a base address.
        // However, it might be a typed SCEVs, so we also have to check if
        // AddressCandidate is in M. If it is, we consider it to be an address
        // in any case. In this case, given that we have traversed
        // AddressCandidate, checkAddressOrTraverse has pushed something on
        // the Worlist, and we have to clear it to prevent the SCEV
        // exploration to continue past the address.
        if (AddressCandidate != Root and M.contains(AddressCandidate)) {
          // We have to stop the search in this direction. Pop back all the
          // nodes that have been pushed by checkAddressOrTraverse.
          for (decltype(NTraversed) I = 0ULL; I < NTraversed; ++I)
            Worklist.pop_back();
          AddrSCEV = AddressCandidate;
        }

      } else {

        // If we have not traversed AddressCandidate, but IsAddr is true. This
        // means that AddressCandidate looks like an address SCEV.
        AddrSCEV = AddressCandidate;

        // Despite the fact that AddressCandidate looks like an address, there
        // are some cases of stuff that looks like an address that should be
        // ignored.
        if (auto *U = dyn_cast<llvm::SCEVUnknown>(AddressCandidate)) {
          auto *UVal = U->getValue();
          if (not isAlwaysAddress(UVal)) {
            // If it's a call there are cases where we know we are never able to
            // say anything meaningful about the type they point to, for now.
            if (auto *Call = dyn_cast<llvm::CallInst>(UVal)) {
              // For OpaqueExtractValue, if they have an aggregate operand that
              // is not a call to an isolated function, we are never able to say
              // anything meaningful about the type they point to, for now.
              // So we just treat them if they are never never addresses that
              // point to a type.
              if (isCallToTagged(Call, FunctionTags::OpaqueExtractValue)) {
                if (not isCallToIsolatedFunction(Call->getOperand(0)))
                  AddrSCEV = nullptr;
              } else {
                // If UVal is a call to a function that was not isolated by
                // revng, the data layout analysis skips it, and we are never
                // able to say something meaningful about the type it points to.
                // So we just treat them if they are never never addresses that
                // point to a type.
                if (not isCallToIsolatedFunction(Call))
                  AddrSCEV = nullptr;
              }
            }
          }
        }
      }
    }

    // Add AddrSCEV to the results if we have found one.
    if (AddrSCEV != nullptr)
      Result.insert(AddrSCEV);
  }

  return Result;
}

size_t
SCEVBaseAddressExplorer::checkAddressOrTraverse(llvm::ScalarEvolution *SE,
                                                const llvm::SCEV *S) {
  auto OldSize = Worklist.size();
  switch (S->getSCEVType()) {

  case llvm::scConstant: {
    // Constants should be checked with isConstantAddress.
    revng_unreachable();
  } break;

  case llvm::scUnknown: {
    // Unknowns are always considered addresses
    llvm::Value *UVal = cast<llvm::SCEVUnknown>(S)->getValue();
    using namespace llvm;
    revng_assert(isa<UndefValue>(UVal) or isa<Argument>(UVal)
                 or isa<Instruction>(UVal) or isa<GlobalVariable>(UVal)
                 or isa<ConstantExpr>(UVal) or isa<ConstantPointerNull>(UVal));
  } break;

  case llvm::scZeroExtend: {
    // Zero extension never changes the value of pointers, so we can safely
    // traversi it.
    const llvm::SCEVZeroExtendExpr *ZE = cast<llvm::SCEVZeroExtendExpr>(S);
    Worklist.push_back(ZE->getOperand());
  } break;

  case llvm::scPtrToInt:
  case llvm::scTruncate:
  case llvm::scSignExtend: {
    // Truncate and Extend are basically casts, so in the first implementation
    // we did not consider them addresses per-se, and we traversed them.
    // So we initially had this code here:
    //    Worklist.push_back(cast<SCEVCastExpr>(S)->getOperand());
    // However, it turned out that tend to show up in nasty situations, so we
    // temporarily disabled their traversal.
    // For now they are simply considered addresses, but we might need to
    // reconsider this behavior in the future, and re-enable their traversal
    // with the two commented lines of code above.
  } break;

  case llvm::scAddExpr: {
    const llvm::SCEVNAryExpr *Add = cast<llvm::SCEVNAryExpr>(S);
    revng_assert(Add->getNumOperands() > 1U);
    // Assume that we only have at most one constant operand.
    bool FoundConstOp = false;
    // Push all the operands on the worlist, to see which of them may be an
    // address.
    for (const llvm::SCEV *Op : Add->operands()) {
      if (auto *C = dyn_cast<llvm::SCEVConstant>(Op)) {
        revng_assert(not FoundConstOp);
        FoundConstOp = true;
        if (not isConstantAddress(C->getValue()))
          continue;
      }
      Worklist.push_back(Op);
    }
  } break;

  case llvm::scAddRecExpr: {
    // We only accept AddRecExpr in certain forms:
    // - the stride must be known
    // - the start must be either a constant, an unknown, a normal add, or
    //   another addrec
    const auto *AddRec = cast<llvm::SCEVAddRecExpr>(S);
    const auto *Start = AddRec->getStart();
    const auto *Stride = AddRec->getStepRecurrence(*SE);
    if (not isa<llvm::SCEVConstant>(Stride)) {
      break;
    }
    // The AddRec is never an address, but we traverse its start expression
    // because it could be an address.
    Worklist.push_back(Start);
  } break;

  case llvm::scMulExpr: {
    // In general scMulExpr are not addresses, so they cannot be traversed.
    // There is only one exception. SCEV do not have the expressive power for
    // representing bit masks. Hence, for expressions of the form
    //   A = B & 0xff00
    // we have a SCEV(A) of the form (B / 8) * 8
    // The SCEV AST tree has the form
    //
    //   B            8
    //    \          /
    //     scUDivExpr      8
    //          \         /
    //           scMulExpr
    //
    // So, we have a MulExpr with exactly two operands, whose second operand
    // is a constant (APConst) that is a power of two, and whose first operand
    // is a UDivExpr (UDiv).
    // (UDiv) also has exactly two operands, and the second is another
    // constant (ADPconstUDiv) which is equal to (APConst). Given that this
    // pattern happens in real code, we pattern match it.
    const llvm::SCEVMulExpr *M = cast<llvm::SCEVMulExpr>(S);
    if (M->getNumOperands() != 2) {
      break;
    }
    const llvm::SCEV *LHS = M->getOperand(0);
    const llvm::SCEV *RHS = M->getOperand(1);
    const auto *LHSConst = dyn_cast<llvm::SCEVConstant>(LHS);
    const auto *RHSConst = dyn_cast<llvm::SCEVConstant>(RHS);
    const auto *LHSUDiv = dyn_cast<llvm::SCEVUDivExpr>(LHS);
    const auto *RHSUDiv = dyn_cast<llvm::SCEVUDivExpr>(RHS);
    const llvm::SCEVConstant *Const = LHSConst ? LHSConst : RHSConst;
    const llvm::SCEVUDivExpr *UDiv = LHSUDiv ? LHSUDiv : RHSUDiv;
    if (Const and UDiv) {
      const llvm::SCEV *UDivRHS = UDiv->getRHS();
      if (const auto *ConstUDivRHS = dyn_cast<llvm::SCEVConstant>(UDivRHS)) {
        const llvm::APInt &APConst = Const->getValue()->getValue();
        const llvm::APInt &APConstUDiv = ConstUDivRHS->getValue()->getValue();
        if (APConst == APConstUDiv and APConst.isPowerOf2()
            and APConstUDiv.isPowerOf2()) {
          // Here we have pattern matched the pattern described above, so we
          // traverse the composite expression A = B & 0xff00 and keep
          // exploring B, without marking A as address.
          const llvm::SCEV *UDivLHS = UDiv->getLHS();
          Worklist.push_back(UDivLHS);
          break;
        }
      }
    }
  } break;

  case llvm::scUDivExpr: {
    // UDivExpr are never addresses unless they are matched as alignment
    // operations along with a MulExpr (see the scMulExpr case).
    // Given that we don't traverse them, if we find a UDivExpr we always
    // consider it an address.
  } break;

  case llvm::scSMaxExpr:
  case llvm::scUMaxExpr:
  case llvm::scSMinExpr:
  case llvm::scUMinExpr:
  case llvm::scSequentialUMinExpr: {
  } break;

  default:
    revng_unreachable("Unknown SCEV kind!");
  }

  auto NewSize = Worklist.size();
  revng_assert(NewSize >= OldSize);
  return NewSize - OldSize;
}
