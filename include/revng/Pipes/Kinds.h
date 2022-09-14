#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::kinds {

template<typename... T>
inline std::tuple<const T &...> fat(const T &...Refs) {
  return std::forward_as_tuple(Refs...);
}

inline pipeline::SingleElementKind Binary("Binary", "A kind to representing the binary as a whole.", ranks::Binary, {}, {});

inline RootKind Root("Root", "The `root` function in the LLVM module.", ranks::Binary);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", "The `root` function in the LLVM module in a version calling isolated functions (see the `Isolated` kind).", Root, ranks::Binary);

inline TaggedFunctionKind
  Isolated("Isolated", "A function in the LLVM module representing "
                                   "an individual function in the binary. In "
                                   "this state, the function has no arguments, "
                                   "return values and still accesses the CPU "
           "state through the CSVs.", ranks::Function, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("ABIEnforced", "A function in the LLVM module "
                                      "representing an individual function in "
                                      "the binary. In this state, the function "
                                      "has arguments and return values "
                                      "according to the the `Prototype` of the "
                                      "corresponding `Function` in the model. "
                                      "The function still accesses the CPU "
              "state through the CSVs.", ranks::Function, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("CSVsPromoted", "Similar to the `ABIEnforced` kind, but "
                                       "the function no longer accesses the "
                                       "CSVs, they are promoted to local "
                                       "variables. These variables are "
                                       "initialized from arguments, if "
               "necessary.", ranks::Function, FunctionTags::CSVsPromoted);

inline pipeline::SingleElementKind Object("Object", "A kind for the object file obtained by compiling "
                                          "the LLVM module containing the `root` function.", ranks::Binary, {}, {});
inline pipeline::SingleElementKind
  Translated("Translated", "A kind for the result of the translation "
                                 "process, representing the whole translated "
             "executable.",ranks::Binary, {}, {});

inline FunctionKind
  FunctionAssemblyInternal("FunctionAssemblyInternal", "A kind representing an internal "
                                             "representation of a disassembled "
                           "function.", ranks::Function, {}, {});

inline FunctionKind FunctionAssemblyPTML("FunctionAssemblyPTML",
                                         "A kind representing the disassembly "
                                         "of a function in PTML form.",
                                         ranks::Function,
                                         fat(ranks::Function,
                                             ranks::BasicBlock,
                                             ranks::Instruction),
                                         {});

inline FunctionKind FunctionControlFlowGraphSVG("FunctionControlFlowGraphSVG",
                                                "A kind representing the SVG "
                                                "image of the control-flow "
                                                "graph of a function.",
                                                ranks::Function,
                                                {},
                                                {});

inline pipeline::SingleElementKind
  BinaryCrossRelations("BinaryCrossRelations", "A kind representing the "
                       "cross-relations of a binary.", ranks::Binary, {}, {});
// WIP
inline pipeline::SingleElementKind
CallGraphSVG("CallGraphSVG", "", ranks::Binary, {}, {});

// WIP
inline FunctionKind
CallGraphSliceSVG("CallGraphSliceSVG", "", ranks::Function, {}, {});

inline constexpr auto BinaryCrossRelationsRole = "CrossRelations";

} // namespace revng::kinds
