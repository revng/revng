#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"

#include "revng/Model/QualifiedType.h"
#include "revng/Model/Segment.h"

/// Use this class to print a graph representing one or more model types
/// and their subtypes.
class TypeSystemPrinter {
private:
  /// Where to print the `.dot` graph
  llvm::raw_ostream &Out;
  /// Visited == all the edges have already been emitted
  llvm::SmallPtrSet<const model::TypeDefinition *, 16> Visited;
  /// Map each Type to the ID of the corresponding emitted `.dot` node
  std::map<const model::TypeDefinition *, uint64_t> NodesMap;

  /// Next free ID to assign to a node
  uint64_t NextID = 0;

public:
  TypeSystemPrinter(llvm::raw_ostream &Out, bool OrthoEdges = false);
  ~TypeSystemPrinter();

private:
  /// Print the given type as a node in a `.dot` graph. This includes the
  /// name, size and all of the fields.
  void dumpTypeNode(const model::TypeDefinition *T, int NodeID);

  /// Add an edge between a field (identified by NodeID+PortID)
  /// and its UnqualifiedType.
  void addFieldEdge(const model::QualifiedType &QT,
                    int SrcID,
                    int SrcPort,
                    int DstID);

private:
  /// Print the given function as a node in a `.dot` graph.
  void dumpFunctionNode(const model::Function &F, int NodeID);

  /// Print the given dynamic function as a node in a `.dot` graph.
  void dumpFunctionNode(const model::DynamicFunction &F, int NodeID);

  /// Print the given segment as a node in a `.dot` graph.
  void dumpSegmentNode(const model::Segment &S, int NodeID);

  /// Add an edge between node_SrcID:SrcPort and node_DstID
  void addEdge(int SrcID, int SrcPort, int DstID);

public:
  /// Generate a graph representation of a given type. Nodes in this graph are
  /// model types, and edges connect fields to their respective UnqualifiedType.
  void print(const model::TypeDefinition &T);

  /// Generate a graph of the types for the given function (Prototype,
  /// StackFrame, ...).
  void print(const model::Function &F);

  /// Generate a graph of the types for the given dynamic function.
  void print(const model::DynamicFunction &F);

  /// Generate a graph of the types for the segment
  void print(const model::Segment &S);

  /// Generate a graph of all the types in a given Module.
  void print(const model::Binary &Model);
};
