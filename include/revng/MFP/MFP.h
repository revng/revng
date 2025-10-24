#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <cstddef>
#include <map>
#include <queue>
#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/ReversePostOrderTraversal.h"

namespace MFP {

inline Logger<> NullLogger("");

template<typename T>
void dump(llvm::raw_ostream &Stream, unsigned Indent, const T &Element) {
  for (unsigned I = 0; I < Indent; ++I)
    Stream << "  ";
  Stream << "(not implemented)\n";
}

template<typename T>
void dumpLabel(llvm::raw_ostream &Stream, const T &Element) {
  Stream << "(not implemented)";
}

template<typename LatticeElement>
struct MFPResult {
  LatticeElement InValue;
  LatticeElement OutValue;
};

/// GT is an instance of llvm::GraphTraits e.g. llvm::GraphTraits<GraphType>
template<typename GT>
auto successors(typename GT::NodeRef From) {
  return llvm::make_range(GT::child_begin(From), GT::child_end(From));
}

template<typename MFI, typename LatticeElement = typename MFI::LatticeElement>
concept MonotoneFrameworkInstance = requires(const MFI &I,
                                             LatticeElement E1,
                                             LatticeElement E2,
                                             typename MFI::Label L) {
  /// To compute the reverse post order traversal of the graph starting from
  /// the extremal nodes, we need that the nodes also represent a subgraph
  typename llvm::GraphTraits<typename MFI::Label>::NodeRef;

  std::same_as<typename MFI::Label,
               typename llvm::GraphTraits<typename MFI::GraphType>::NodeRef>;
  { I.combineValues(E1, E2) } -> std::same_as<LatticeElement>;
  { I.isLessOrEqual(E1, E2) } -> std::same_as<bool>;
  { I.applyTransferFunction(L, E2) } -> std::same_as<LatticeElement>;
};

template<typename Label, typename LatticeElement>
using ResultMap = std::map<Label, MFPResult<LatticeElement>>;

template<MonotoneFrameworkInstance MFI>
using MFIResultMap = ResultMap<typename MFI::Label,
                               typename MFI::LatticeElement>;

/// Compute the maximum fixed points of an instance of monotone framework GT an
/// instance of llvm::GraphTraits that tells us how to visit the graph LGT a
/// graph type that tells us how to visit the subgraph induced by a node in the
/// graph. This is needed for the RPOT because for certain graph (e.g.
/// Inverse<...>) the nodes don't necessary carry all the information that
/// GraphType has.
template<MonotoneFrameworkInstance MFIType,
         typename GT = llvm::GraphTraits<typename MFIType::GraphType>,
         typename LGT = typename MFIType::Label>
MFIResultMap<MFIType>
getMaximalFixedPoint(const MFIType &MFI,
                     typename MFIType::GraphType Flow,
                     typename MFIType::LatticeElement InitialValue,
                     typename MFIType::LatticeElement ExtremalValue,
                     const std::vector<typename MFIType::Label> &ExtremalLabels,
                     const std::vector<typename MFIType::Label> &InitialNodes,
                     Logger<> &Logger = NullLogger) {
  using Label = typename MFIType::Label;
  using LatticeElement = typename MFIType::LatticeElement;

  std::map<Label, LatticeElement> PartialAnalysis;
  std::map<Label, MFPResult<LatticeElement>> AnalysisResult;

  struct WorklistItem {
    size_t Priority;
    Label Item;

    std::weak_ordering operator<=>(const WorklistItem &) const = default;
  };
  std::set<WorklistItem> Worklist;

  llvm::SmallSet<Label, 8> Visited{};
  std::map<Label, size_t> LabelPriority;

  //
  // Initialize the worklist and extremal labels
  //

  if (Logger.isEnabled()) {
    revng_log(Logger, "Initializing extremal labels");
    LoggerIndent<> Indent(Logger);
    Logger << "Extremal value:\n";
    MFP::dump(*Logger.getAsLLVMStream(), 1, ExtremalValue);
    Logger << DoLog;

    Logger << "Extremal labels:" << DoLog;
    LoggerIndent<> Indent2(Logger);
    for (Label ExtremalLabel : ExtremalLabels) {
      MFP::dumpLabel(*Logger.getAsLLVMStream(), ExtremalLabel);
      Logger << DoLog;
    }
  }

  for (Label ExtremalLabel : ExtremalLabels)
    AnalysisResult[ExtremalLabel].InValue = ExtremalValue;

  if (Logger.isEnabled()) {
    revng_log(Logger, "Initializing initial nodes");
    LoggerIndent<> Indent(Logger);
    Logger << "Initial value:\n";
    MFP::dump(*Logger.getAsLLVMStream(), 1, InitialValue);
    Logger << DoLog;

    Logger << "Initial labels:" << DoLog;
    LoggerIndent<> Indent2(Logger);
    for (Label InitialNode : InitialNodes) {
      MFP::dumpLabel(*Logger.getAsLLVMStream(), InitialNode);
      Logger << DoLog;
    }
  }

  for (Label Start : InitialNodes) {

    if (Visited.contains(Start))
      continue;

    // Fill the worklist with nodes in reverse post order launching a visit
    // from each remaining node
    ReversePostOrderTraversalExt<LGT, GT, llvm::SmallSet<Label, 8>>
      RPOTE(Start, Visited);
    for (Label Node : RPOTE) {
      LabelPriority[Node] = LabelPriority.size();
      Worklist.insert({ LabelPriority.at(Node), Node });

      // Initialize the analysis value for non extremal nodes
      if (!AnalysisResult.contains(Node))
        AnalysisResult[Node].InValue = InitialValue;
    }
  }

  // Step 2 iterations
  revng_log(Logger, "Starting the iterations");
  LoggerIndent<> Indent(Logger);

  unsigned IterationIndex = 0;
  while (not Worklist.empty()) {
    // Fetch the next label from the worklist
    WorklistItem First = *Worklist.begin();
    Label Start = First.Item;
    Worklist.erase(First);

    auto &LabelAnalysis = AnalysisResult.at(Start);

    if (Logger.isEnabled()) {
      Logger << "Iteration #" << IterationIndex << " on ";
      MFP::dumpLabel(*Logger.getAsLLVMStream(), Start);
      Logger << DoLog;
    }

    LoggerIndent<> Indent(Logger);

    if (Logger.isEnabled()) {
      Logger << "Initial value:\n";
      MFP::dump(*Logger.getAsLLVMStream(), 1, LabelAnalysis.InValue);
      Logger << DoLog;

      Logger << "Final value:\n";
      MFP::dump(*Logger.getAsLLVMStream(), 1, LabelAnalysis.OutValue);
      Logger << DoLog;
    }

    // Run the transfer function
    revng_log(Logger, "Running the transfer function");
    Logger.indent();
    const auto &New = MFI.applyTransferFunction(Start, LabelAnalysis.InValue);
    Logger.unindent();

    if (Logger.isEnabled()) {
      LoggerIndent<> Indent(Logger);
      Logger << "New final value:\n";
      MFP::dump(*Logger.getAsLLVMStream(), 1, New);
      Logger << DoLog;
    }

    // TODO: assert that MFI.isLessOrEqual(LabelAnalysis.OutValue, New)

    // Save the new final value
    LabelAnalysis.OutValue = New;

    // Enqueue successors that need to be recomputed
    revng_log(Logger, "Processing successors:");
    LoggerIndent<> Indent2(Logger);
    for (Label Successor : successors<GT>(Start)) {
      auto &SuccessorResults = AnalysisResult.at(Successor);

      if (Logger.isEnabled()) {
        Logger << "Considering successor ";
        MFP::dumpLabel(*Logger.getAsLLVMStream(), Successor);
        Logger << DoLog;

        Logger << "Initial value:\n";
        LoggerIndent<> Indent(Logger);
        Logger << DoLog;
        MFP::dump(*Logger.getAsLLVMStream(), 1, SuccessorResults.InValue);
        Logger << DoLog;
      }
      LoggerIndent<> Indent(Logger);

      if (not MFI.isLessOrEqual(LabelAnalysis.OutValue,
                                SuccessorResults.InValue)) {
        // We need to re-enqueue

        // Combine the old value with the new incoming value and update it
        SuccessorResults.InValue = MFI.combineValues(SuccessorResults.InValue,
                                                     LabelAnalysis.OutValue);

        if (Logger.isEnabled()) {
          Logger << "Enqueuing. New initial value:\n";
          MFP::dump(*Logger.getAsLLVMStream(), 1, SuccessorResults.InValue);
          Logger << DoLog;
        }

        // Enqueue in the list
        Worklist.insert({ LabelPriority.at(Successor), Successor });
      } else {
        revng_log(Logger, "Ignoring");
      }
    }

    ++IterationIndex;
  }

  return AnalysisResult;
}

template<MonotoneFrameworkInstance MFI,
         typename GT = llvm::GraphTraits<typename MFI::GraphType>,
         typename LGT = typename MFI::Label>
MFIResultMap<MFI>
getMaximalFixedPoint(const MFI &Instance,
                     typename MFI::GraphType Flow,
                     typename MFI::LatticeElement InitialValue,
                     typename MFI::LatticeElement ExtremalValue,
                     const std::vector<typename MFI::Label> &ExtremalLabels,
                     Logger<> &Logger = NullLogger) {
  using Label = typename MFI::Label;
  std::vector<Label> InitialNodes(ExtremalLabels);

  // Handle the special case that the graph has a single entry node
  if (GT::getEntryNode(Flow) != typename GT::NodeRef{}) {
    InitialNodes.push_back(GT::getEntryNode(Flow));
  }
  // Start visits for nodes that we still haven't visited
  // prioritizing extremal nodes
  for (Label Node :
       llvm::make_range(GT::nodes_begin(Flow), GT::nodes_end(Flow))) {
    InitialNodes.push_back(Node);
  }
  return getMaximalFixedPoint<MFI, GT, LGT>(Instance,
                                            Flow,
                                            InitialValue,
                                            ExtremalValue,
                                            ExtremalLabels,
                                            InitialNodes,
                                            Logger);
}

} // namespace MFP
