#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//
#include <variant>

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Assert.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

namespace vma {

// --------------- Node content

/// The atomic element to which we want to attach type information
using UseOrValue = std::variant<const llvm::Use *, const llvm::Value *>;

inline bool isValue(const UseOrValue &Content) {
  return std::holds_alternative<const llvm::Value *>(Content);
};

inline bool isUse(const UseOrValue &Content) {
  return std::holds_alternative<const llvm::Use *>(Content);
};

inline const llvm::Value *getValue(const UseOrValue &Content) {
  revng_assert(isValue(Content));
  return std::get<const llvm::Value *>(Content);
};

inline const llvm::Use *getUse(const UseOrValue &Content) {
  revng_assert(isUse(Content));
  return std::get<const llvm::Use *>(Content);
};

/// Check if the variant is holding an Instruction's output Value
inline bool isInst(const UseOrValue &Content) {
  return isValue(Content) && isa<llvm::Instruction>(getValue(Content));
}

/// Get the index of an operand inside its Instruction
inline unsigned getOpNo(const UseOrValue &Content) {
  revng_assert(isUse(Content));
  return getUse(Content)->getOperandNo();
};

// --------------- Node Color initialization

/// Represents the initial color of a node and the ones it can accept
struct NodeColorProperty {
  const ColorSet InitialColor = NO_COLOR;
  const ColorSet AcceptedColors = NO_COLOR;

  NodeColorProperty(ColorSet Initial, ColorSet Accepted) :
    InitialColor(Initial), AcceptedColors(Accepted) {}
};

/// Return the type colors associated to a given Use or Value
NodeColorProperty nodeColors(const UseOrValue &NC);

// --------------- TypeFlowGraph Node

/// Label of a TypeFlowGraph edge
///
/// Indicates which colors can be propagated from the source node to the target
/// node of an edge on the TypeFlowGraph.
struct EdgeLabel {
  ColorSet Colors;

  EdgeLabel() : Colors(NO_COLOR) {}
  EdgeLabel(ColorSet C) : Colors(C) {}
  EdgeLabel(unsigned U) : Colors(U) {}
};

/// Node data containing colors for an `llvm::Use` or `llvm::Value`
class TypeFlowNodeData {
private:
  /// LLVM Use or Value to which the type information is attached
  const UseOrValue Content;
  /// Candidate types for this node
  ColorSet Candidates;
  /// Types which this node can be infected with
  const ColorSet Accepted;

public:
  TypeFlowNodeData() = delete;
  TypeFlowNodeData(const UseOrValue &NC, const NodeColorProperty Colors) :
    Content(NC),
    Candidates(Colors.InitialColor),
    Accepted(Colors.AcceptedColors) {}

  TypeFlowNodeData(const TypeFlowNodeData &N) = default;
  TypeFlowNodeData(TypeFlowNodeData &&N) = default;
  TypeFlowNodeData &operator=(const TypeFlowNodeData &N) = delete;
  TypeFlowNodeData &operator=(TypeFlowNodeData &&N) = delete;

public:
  UseOrValue getContent() const { return Content; }
  bool isUse() const { return vma::isUse(Content); }
  bool isValue() const { return vma::isValue(Content); }
  const llvm::Use *getUse() const { return vma::getUse(Content); }
  const llvm::Value *getValue() const { return vma::getValue(Content); }

  /// Has no candidate color
  bool isUncolored() const { return Candidates.countValid() == 0; }
  /// Has exactly one candidate color
  bool isDecided() const { return Candidates.countValid() == 1; }
  /// Has more than one candidate color
  bool isUndecided() const { return Candidates.countValid() > 1; }

  ColorSet getCandidates() const { return Candidates; }
  void setCandidates(ColorSet Color) {
    revng_assert(Accepted.contains(Color));
    Candidates = Color;
  }

  ColorSet getAccepted() const { return Accepted; }

public:
  /// Print a textual representation of the node's content
  void print(llvm::raw_ostream &Out) const debug_function;

  /// Print the content of the node to a string
  std::string toString() const debug_function;
};

/// Add GenericGraph's BidirectionalNode interface
using TypeFlowNode = BidirectionalNode<TypeFlowNodeData, EdgeLabel, false>;

} // namespace vma
