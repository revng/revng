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

/// A dynamic hierarcy is a tree that can be extended by downstream
/// libraries. By instantiating a node and providing a parent then the child
/// will be visible from the parent as well.
///
/// \note init must be called after every object of the hierarchy has been
///       constructed.
template<typename DerivedType>
class DynamicHierarchy {
public:
  using entry_t = size_t;

private:
  static inline std::atomic<bool> Initialized = false;

private:
  std::vector<DynamicHierarchy *> Children;
  DynamicHierarchy *Parent;
  entry_t Start;
  entry_t End;
  std::string Name;

public:
  DynamicHierarchy(llvm::StringRef Name) : Parent(nullptr), Name(Name.str()) {
    getRoots().push_back(&self());
    getAll().push_back(&self());

    // NOTE: order the entries, in order to guarantee consistent ordering
    // irrespective of load order
    llvm::sort(getRoots(), compareByName);
    llvm::sort(getAll(), compareByName);
  }

  DynamicHierarchy(llvm::StringRef Name, DynamicHierarchy &Parent) :
    Parent(&Parent), Name(Name.str()) {
    getAll().push_back(&self());

    // NOTE: see constructor above
    llvm::sort(getAll(), compareByName);
  }

  DynamicHierarchy(DynamicHierarchy &&) = delete;
  DynamicHierarchy(const DynamicHierarchy &) = delete;
  DynamicHierarchy &operator=(DynamicHierarchy &&) = delete;
  DynamicHierarchy &operator=(const DynamicHierarchy &) = delete;

  ~DynamicHierarchy() = default;

public:
  static void init() {
    bool ExpectedInitialized = false;
    if (atomic_compare_exchange_weak(&Initialized,
                                     &ExpectedInitialized,
                                     true)) {
      for (DynamicHierarchy *Root : getAll())
        Root->registerInParent();
      static entry_t ID = -1;
      for (DynamicHierarchy *Root : getRoots())
        ID = Root->assign(ID);
    }
  }

  static std::vector<DerivedType *> &getAll() {
    static std::vector<DerivedType *> AllNodes;
    return AllNodes;
  }

  static std::vector<DerivedType *> &getRoots() {
    static std::vector<DerivedType *> Roots;
    return Roots;
  }

  static DerivedType *findByName(llvm::StringRef Name) {
    DerivedType *Result = nullptr;
    for (DerivedType *Node : getAll()) {
      if (Node->name() == Name) {
        revng_assert(Result == nullptr);
        Result = Node;
      }
    }

    return Result;
  }

public:
  llvm::ArrayRef<DynamicHierarchy *> children() const { return Children; }

  DerivedType *parent() {
    if (not Parent)
      return nullptr;
    return &Parent->self();
  }

  entry_t id() const {
    init();
    return Start;
  }

  llvm::StringRef name() const { return Name; }

public:
  bool isa(entry_t ID) const { return id() == ID; }

  bool ancestorOf(const DynamicHierarchy &MaybeChild) const {
    return ancestorOf(MaybeChild.id());
  }

  bool ancestorOf(entry_t ID) const { return Start <= ID and ID < End; }

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

  size_t depth() const {
    size_t ToReturn = 0;
    auto Current = Parent;
    while (Current != nullptr) {
      ToReturn++;
      Current = Current->Parent;
    }
    return ToReturn;
  }

public:
  void dump() const debug_function { dump(dbg, 0); }

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

private:
  void registerInParent() {
    if (Parent != nullptr)
      Parent->Children.push_back(this);
  }

  entry_t assign(entry_t ID = -1) {

    Start = ++ID;

    for (DynamicHierarchy *Child : Children) {
      ID = Child->assign(ID);
    }

    End = ID + 1;

    return ID;
  }

private:
  const DerivedType &self() const { return *static_cast<DerivedType *>(this); }
  DerivedType &self() { return *static_cast<DerivedType *>(this); }

  static bool compareByName(DerivedType *Elem1, DerivedType *Elem2) {
    revng_assert(Elem1->name() != Elem2->name() or Elem1 == Elem2,
                 ("There are two dynamic hierarchy elements named "
                  + Elem1->name().str())
                   .c_str());
    return Elem1->name() < Elem2->name();
  }
};
