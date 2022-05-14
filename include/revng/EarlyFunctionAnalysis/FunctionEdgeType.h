#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: FunctionEdgeType
doc: Type of edge on the CFG
type: enum
members:
  - name: DirectBranch
    doc: Branch due to function-local CFG (a regular branch)
  - name: FakeFunctionCall
    doc: A call to a fake function
  - name: FakeFunctionReturn
    doc: A return from a fake function
  - name: FunctionCall
    doc: A function call for which the cache was able to produce a summary
  - name: IndirectCall
    doc: A function call for which the target is unknown
  - name: Return
    doc: A proper function return
  - name: BrokenReturn
    doc: |
      A branch returning to the return address, but leaving the stack in an
      unexpected situation
  - name: IndirectTailCall
    doc: A branch representing an indirect tail call
  - name: LongJmp
    doc: A branch representing a longjmp or similar constructs
  - name: Killer
    doc: A killer basic block (killer syscall or endless loop)
  - name: Unreachable
    doc: The basic block ends with an unreachable instruction
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionEdgeType.h"

// TODO: we need to handle noreturn function calls

namespace efa::FunctionEdgeType {

inline bool isCall(Values V) {
  switch (V) {
  case Count:
    revng_abort();
    break;

  case FunctionCall:
  case IndirectCall:
  case IndirectTailCall:
    return true;

  case Invalid:
  case DirectBranch:
  case FakeFunctionCall:
  case FakeFunctionReturn:
  case Return:
  case BrokenReturn:
  case LongJmp:
  case Killer:
  case Unreachable:
    return false;
  }
}

inline bool needsFallthrough(Values V) {
  switch (V) {
  case FunctionCall:
  case IndirectCall:
    return true;

  case DirectBranch:
  case FakeFunctionCall:
  case FakeFunctionReturn:
  case IndirectTailCall:
  case Return:
  case BrokenReturn:
  case LongJmp:
  case Killer:
  case Unreachable:
    return false;

  case Invalid:
  case Count:
    revng_abort();
    break;
  }
}

} // namespace efa::FunctionEdgeType

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdgeType.h"
