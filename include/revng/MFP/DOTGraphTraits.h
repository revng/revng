#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/DOTGraphTraits.h"

#include "revng/MFP/Graph.h"
#include "revng/Support/DOTGraphTraits.h"

namespace llvm {
class Instruction;
}

template<MFP::MonotoneFrameworkInstance MFI>
struct llvm::DOTGraphTraits<MFP::Graph<MFI> *>
  : public llvm::DefaultDOTGraphTraits {

  static_assert(MFP::SerializableLatticeElement<typename MFI::LatticeElement>);

  using GraphType = MFP::Graph<MFI> *;
  using UnderlyingGraphType = MFI::GraphType;
  using UnderlyingDOTGraphTraits = llvm::DOTGraphTraits<UnderlyingGraphType>;
  using NodeRef = llvm::GraphTraits<GraphType>::NodeRef;

  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const GraphType &G) {
    return UnderlyingDOTGraphTraits::getGraphName(G->underlying());
  }

  static std::string getGraphProperties(const GraphType &G) {
    return UnderlyingDOTGraphTraits::getGraphProperties(G->underlying());
  }

  static bool renderGraphFromBottomUp() {
    return UnderlyingDOTGraphTraits::renderGraphFromBottomUp();
  }

  static bool isNodeHidden(const NodeRef &Arg, const GraphType &G) {
    return UnderlyingDOTGraphTraits::isNodeHidden(Arg, G->underlying());
  }

  static bool renderNodesUsingHTML() {
    return UnderlyingDOTGraphTraits::renderNodesUsingHTML();
  }

  std::string getNodeLabel(const NodeRef &Arg, const GraphType &G) {
    std::string Result;

    std::string Newline = renderNodesUsingHTML() ? HTMLNewline : "\\l";

    {
      llvm::raw_string_ostream Stream(Result);
      auto AnalysisResult = G->results().at(Arg);

      // Dump underlying graph node label
      Stream << UnderlyingDOTGraphTraits::getNodeLabel(Arg, G->underlying());
      Stream << Newline << Newline;

      auto Dump = [&Newline](const auto &ToDump, StringRef Name) {
        std::string Result;
        {
          llvm::raw_string_ostream Stream(Result);
          Stream << Name.str() << " value:"
                 << "\n";
          MFP::dump(Stream, 1, ToDump);
        }

        replaceAll(Result, "\n", Newline);

        return Result;
      };

      // Dump initial value
      Stream << "<FONT FACE=\"monospace\">";
      Stream << Dump(AnalysisResult.InValue, "Initial");
      Stream << Newline;
      Stream << Dump(AnalysisResult.OutValue, "Final");
      Stream << "</FONT>";
    }

    return Result;
  }

  static std::string
  getNodeIdentifierLabel(const NodeRef &Arg, const GraphType &G) {
    return UnderlyingDOTGraphTraits::getNodeIdentifierLabel(Arg,
                                                            G->underlying());
  }

  static std::string
  getNodeDescription(const NodeRef &Arg, const GraphType &G) {
    return UnderlyingDOTGraphTraits::getNodeDescription(Arg, G->underlying());
  }

  static std::string getNodeAttributes(const NodeRef &Arg, const GraphType &G) {
    return UnderlyingDOTGraphTraits::getNodeAttributes(Arg, G->underlying());
  }

  template<typename EdgeIter>
  static std::string
  getEdgeAttributes(const NodeRef &Arg, EdgeIter It, GraphType G) {
    return UnderlyingDOTGraphTraits::getEdgeAttributes(Arg,
                                                       It,
                                                       G->underlying());
  }

  template<typename EdgeIter>
  static std::string getEdgeSourceLabel(const NodeRef &Arg, EdgeIter It) {
    return UnderlyingDOTGraphTraits::getEdgeSourceLabel(Arg, It);
  }

  template<typename EdgeIter>
  static bool edgeTargetsEdgeSource(const NodeRef &Arg, EdgeIter It) {
    return UnderlyingDOTGraphTraits::edgeTargetsEdgeSource(Arg, It);
  }

  template<typename EdgeIter>
  static EdgeIter getEdgeTarget(const NodeRef &Arg, EdgeIter It) {
    return UnderlyingDOTGraphTraits::getEdgeTarget(Arg, It);
  }

  static bool hasEdgeDestLabels() {
    return UnderlyingDOTGraphTraits::hasEdgeDestLabels();
  }

  static unsigned numEdgeDestLabels(const NodeRef &Arg) {
    return UnderlyingDOTGraphTraits::numEdgeDestLabels(Arg);
  }

  static std::string getEdgeDestLabel(const NodeRef &Arg1, unsigned Arg2) {
    return UnderlyingDOTGraphTraits::getEdgeDestLabel(Arg1, Arg2);
  }

  template<typename GraphWriter>
  static void addCustomGraphFeatures(GraphType G, GraphWriter &Writer) {
    UnderlyingDOTGraphTraits::addCustomGraphFeatures(G->underlying(), Writer);
  }
};
