#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace llvm {

class Instruction;
class Value;

} // namespace llvm

namespace revng {

/// True if the clifter emits no Clift operation for \p I: the scope-graph
/// markers, which only carry the structure of the scope graph.
///
/// This is about the instruction, not its effect. A goto marker emits nothing
/// itself, but still makes the terminator of its block a `goto` rather than a
/// fallthrough into the successor's scope.
bool isNotEmitted(const llvm::Instruction &I);

/// True if the clifter emits \p I as a C statement of its own: a terminator, a
/// local variable declaration, or the root of an expression tree. False for
/// anything inlined into its users, and false for \ref isNotEmitted, which is
/// neither. Callers walking a basic block skip those first.
bool isStatement(const llvm::Instruction &I);

/// True if a use of \p V costs a single Clift node, because the clifter emits
/// a reference to something that exists on its own: an argument, a constant, a
/// global, or a \ref isStatement instruction, which by then has storage.
bool isExpressionLeaf(const llvm::Value &V);

} // namespace revng
