#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iostream>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Debug.h"

/// a dynamic hierarcy is a tree that can be extended by
/// downstream libraries. By instantiating a node and providing a
/// parent then the child will be visible from the parent as well.
///
/// notice that init must be called after every object of the hierarchy has been
/// constructed.
template<typename DerivedType>
class DynamicHierarchy {
public:
  using entry_t = size_t;

  DynamicHierarchy(llvm::StringRef Name) : Name(Name.str()) {
    getRoots().push_back(&self());
  }
  DynamicHierarchy(llvm::StringRef Name, DynamicHierarchy &Parent) :
    Name(Name.str()) {
    Parent.Children.push_back(this);
  }

  DynamicHierarchy(DynamicHierarchy &&) = delete;
  DynamicHierarchy(const DynamicHierarchy &) = delete;
  DynamicHierarchy &operator=(DynamicHierarchy &&) = delete;
  DynamicHierarchy &operator=(const DynamicHierarchy &) = delete;
  ~DynamicHierarchy() = default;

  bool isa(entry_t ID) const { return id() == ID; }

  bool ancestorOf(const DynamicHierarchy &MaybeChild) const {
    return ancestorOf(MaybeChild.id());
  }

  bool ancestorOf(entry_t ID) const { return Start <= ID and ID < End; }
  entry_t id() const { return Start; }

  DerivedType *parent() {
    if (not Parent)
      return nullptr;
    return &Parent->self();
  }
  const DerivedType *parent() const {
    if (not Parent)
      return nullptr;
    return &Parent->self();
  }

  DerivedType *getRootAncestor() {
    DynamicHierarchy *LastAncestor = this;
    DynamicHierarchy *NextAncestor = Parent;
    while (NextAncestor != nullptr) {
      LastAncestor = NextAncestor;
      NextAncestor = NextAncestor->parent();
    }
    return &LastAncestor->self();
  }

  const DerivedType *getRootAncestor() const {
    DynamicHierarchy *LastAncestor = this;
    DynamicHierarchy *NextAncestor = Parent;
    while (NextAncestor != nullptr) {
      LastAncestor = NextAncestor;
      NextAncestor = NextAncestor->parent();
    }
    return &LastAncestor->self();
  }

  void dump() const { dump(dbg, 0); }

  template<typename OS>
  void dump(OS &Output, size_t Indent = 0) const {
    for (size_t I = 0; I < Indent; ++I)
      Output << "  ";
    Output << "Start " << Start << "\n";

    for (DynamicHierarchy *Child : Children) {
      Child->dump<OS>(Output, Indent + 1);
    }

    for (size_t I = 0; I < Indent; ++I)
      Output << "  ";
    Output << "End " << End << "\n";
  }

  llvm::StringRef name() const { return Name; }

  size_t depth() const {
    size_t ToReturn = 0;
    auto Current = Parent;
    while (Current != nullptr) {
      ToReturn++;
      Current = Current->Parent;
    }
    return ToReturn;
  }

  llvm::ArrayRef<DynamicHierarchy *> children() const { return Children; }

  static void init() {
    static entry_t ID = -1;
    for (DynamicHierarchy *Root : getRoots())
      ID = Root->assign(ID);
  }

  static std::vector<DerivedType *> &getRoots() {
    static std::vector<DerivedType *> Roots;
    return Roots;
  }

private:
  entry_t assign(entry_t ID = -1) {

    Start = ++ID;

    for (DynamicHierarchy *Child : Children) {
      ID = Child->assign(ID);
      Child->Parent = this;
    }

    End = ID + 1;

    return ID;
  }

  const DerivedType &self() const { return *static_cast<DerivedType *>(this); }
  DerivedType &self() { return *static_cast<DerivedType *>(this); }

  std::vector<DynamicHierarchy *> Children;
  DynamicHierarchy *Parent;
  entry_t Start;
  entry_t End;
  std::string Name;
};
