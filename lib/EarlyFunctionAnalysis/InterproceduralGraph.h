#pragma once

/// \file InterproceduralGraph.h

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/IR/BasicBlock.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/MetaAddress.h"

namespace efa {

struct Node {
  enum class AnalysisType {
    UAOF,
    RAOFC,
    URVOF,
    URVOFC,
  };

  enum class ResultType { Arguments, Returns };

  using FunctionAddress = MetaAddress;
  struct CallSiteAddress {
    MetaAddress Entry;
    MetaAddress CallSite;

    std::strong_ordering
    operator<=>(const CallSiteAddress &RHS) const = default;
  };
  using AddressType = std::variant<FunctionAddress, CallSiteAddress>;

  /// If Address is invalid, it means Node represents Call Site for indirect
  /// call
  AddressType Address;

  /// BB is either Entry basic block for function nodes or Call Site basic block
  /// for call sites.
  const llvm::BasicBlock *BB;

  /// Type of node can be either Result or Analysis. std::monostate is reserved
  /// for entry node
  using VType = std::variant<std::monostate, AnalysisType, ResultType>;
  VType Type;

  explicit Node() : Address{}, Type{ std::monostate{} } {}

  explicit Node(AddressType Address, const llvm::BasicBlock *BB, VType V) :
    Address{ Address }, BB{ BB }, Type{ V } {}

  bool isFunction() const {
    return std::holds_alternative<FunctionAddress>(Address);
  }

  bool isCallSite() const {
    return std::holds_alternative<CallSiteAddress>(Address);
  }

  FunctionAddress getFunctionAddress() const {
    if (isFunction()) {
      return std::get<FunctionAddress>(Address);
    } else {
      return getCallSiteAddresses().Entry;
    }
  }

  CallSiteAddress getCallSiteAddresses() const {
    revng_assert(isCallSite());
    return std::get<CallSiteAddress>(Address);
  }

  bool isIndirect() const {
    if (isFunction()) {
      return getFunctionAddress().isInvalid();
    }

    return false;
  }

  bool isEntry() const { return std::holds_alternative<std::monostate>(Type); }

  bool isAnalysis() const { return std::holds_alternative<AnalysisType>(Type); }

  bool isResult() const { return std::holds_alternative<ResultType>(Type); }

  ResultType getResultType() const {
    revng_assert(isResult());
    return std::get<ResultType>(Type);
  }

  AnalysisType getAnalysisType() const {
    revng_assert(isAnalysis());
    return std::get<AnalysisType>(Type);
  }

  std::string getLabel() const {
    const auto
      AddressStr = (isFunction()) ?
                     getFunctionAddress().toString() :
                     llvm::formatv("{0}, {1}",
                                   getCallSiteAddresses().Entry.toString(),
                                   getCallSiteAddresses().CallSite.toString());
    const auto Str = (!isIndirect()) ?
                       AddressStr :
                       llvm::formatv("Indirect_{0}",
                                     reinterpret_cast<const void *>(BB))
                         .str();
    if (isAnalysis()) {
      switch (std::get<AnalysisType>(Type)) {
      case AnalysisType::UAOF:
        return llvm::formatv("UAOF({0}))", Str);
      case AnalysisType::URVOF:
        return llvm::formatv("URVOF({0})", Str);
      case AnalysisType::RAOFC:
        return llvm::formatv("RAOFC({0})", Str);
      case AnalysisType::URVOFC:
        return llvm::formatv("URVOFC({0})", Str);
      }
    } else if (std::holds_alternative<ResultType>(Type)) {
      if (std::get<ResultType>(Type) == ResultType::Arguments) {
        return llvm::formatv("Arguments({0})", Str);
      } else {
        return llvm::formatv("Returns({0})", Str);
      }
    } else {
      return "EntryNode";
    }
  }

  std::string getAttributes() const {
    if (isAnalysis()) {
      auto Analysis = getAnalysisType();
      if (Analysis == AnalysisType::RAOFC || Analysis == AnalysisType::URVOFC) {
        return "color=blue";
      } else {
        return "color=red";
      }
    } else if (isResult()) {
      return "shape=tab";
    } else {
      return "shape=ellipse, color=green";
    }
  }
};

using InterproceduralNode = ForwardNode<Node>;

struct InterproceduralGraph : GenericGraph<InterproceduralNode> {};

} // namespace efa

template<>
class llvm::DOTGraphTraits<efa::InterproceduralGraph *>
  : public llvm::DefaultDOTGraphTraits {
public:
  DOTGraphTraits(bool Simple = false) : llvm::DefaultDOTGraphTraits(Simple) {}

  static std::string getNodeLabel(const efa::InterproceduralNode *P,
                                  const efa::InterproceduralGraph *G) {
    return P->getLabel();
  }

  static std::string getNodeAttributes(const efa::InterproceduralNode *P,
                                       const efa::InterproceduralGraph *G) {
    return P->getAttributes();
  }
};
