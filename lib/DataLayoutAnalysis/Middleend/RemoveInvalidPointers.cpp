//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "DLAStep.h"

bool dla::RemoveInvalidPointers::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  for (LayoutTypeSystemNode *PtrNode : llvm::nodes(&TS)) {

    // Nodes whose size is the same of a pointer cannot have invalid outgoing
    // pointer edges.
    if (PtrNode->Size == PointerSize)
      continue;

    auto It = PtrNode->Successors.begin();
    auto End = PtrNode->Successors.end();
    auto Next = End;
    for (; It != End; It = Next) {

      Next = std::next(It);

      // Ignore non-pointer edges
      if (not isPointerEdge(*It))
        continue;

      // If we reach this point PtrNode->Size is different from a pointer size
      // and It points to a pointer edge that is outgoing from PtrNode.
      // That edge is invalid, because according to PtrNode->Size PtrNode cannot
      // be a pointer, so we remove the wrong edge.
      LayoutTypeSystemNode *Pointee = It->first;
      auto PredIt = Pointee->Predecessors.find({ PtrNode, It->second });
      revng_assert(PredIt != Pointee->Predecessors.end());
      Pointee->Predecessors.erase(PredIt);
      PtrNode->Successors.erase(It);
      Changed = true;
    }
  }

  return Changed;
}
