#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <utility>

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::clift {

template<typename StorageT,
         typename BaseT,
         typename SubobjectT,
         typename ValueT = std::monostate>
class ClassTypeStorage : public BaseT {
  struct KeyBase {
    llvm::StringRef Name;
    llvm::ArrayRef<SubobjectT> Subobjects;
  };

  struct KeyDefinition : KeyBase, ValueT {
    template<typename... ArgsT>
    explicit KeyDefinition(mlir::StorageUniquer::StorageAllocator &Allocator,
                           const llvm::StringRef Name,
                           const llvm::ArrayRef<SubobjectT> Subobjects,
                           ArgsT &&...Args) :
      KeyBase{ Allocator.copyInto(Name), Allocator.copyInto(Subobjects) },
      ValueT{ std::forward<ArgsT>(Args)... } {}

    explicit KeyDefinition(mlir::StorageUniquer::StorageAllocator &Allocator,
                           const KeyDefinition &Key) :
      KeyBase{ Allocator.copyInto(Key.Name),
               Allocator.copyInto(Key.Subobjects) },
      ValueT(static_cast<const ValueT &>(Key)) {}
  };

  // Some internal mechanism of mlir have access to the key of the storage
  // instead of the entire storage or the key using that storage. For that
  // reason it is best to throw everything related to the type inside the key,
  // and then override operators to pretend they don't know about the non key
  // fields. That way when it is needed one can access everything.
  struct Key {
    uint64_t ID;
    std::optional<KeyDefinition> Definition;

    Key(const uint64_t ID) : ID(ID) {}

    // struct storages are never exposed to the user, they are only used
    // internally to figure out how to create unique objects. only operator== is
    // every used, everything else is handled by hasValue
    friend bool operator==(const Key &LHS, const Key &RHS) {
      return LHS.ID == RHS.ID;
    }

    [[nodiscard]] llvm::hash_code hashValue() const {
      return llvm::hash_value(ID);
    }
  };

  Key TheKey;

public:
  using KeyTy = Key;

  using SubobjectTy = SubobjectT;

  const Key &getAsKey() const { return TheKey; }

  static llvm::hash_code hashKey(const KeyTy &Key) { return Key.hashValue(); }

  bool operator==(const Key &Other) const { return TheKey == Other; }

  template<typename... ArgsT>
  ClassTypeStorage(const uint64_t ID) : TheKey(ID) {}

  static StorageT *construct(mlir::StorageUniquer::StorageAllocator &Allocator,
                             const KeyTy &Key) {
    StorageT *const S = new (Allocator.allocate<StorageT>()) StorageT(Key.ID);
    if (Key.Definition)
      S->TheKey.Definition.emplace(Allocator, *Key.Definition);
    return S;
  }

  /// Define a mutation method for changing the type after it is created. In
  /// many cases, we only want to set the mutable component once and reject
  /// any further modification, which can be achieved by returning failure
  /// from this function.
  template<typename... ArgsT>
  [[nodiscard]] mlir::LogicalResult
  mutate(mlir::StorageUniquer::StorageAllocator &Allocator,
         const llvm::StringRef Name,
         const llvm::ArrayRef<SubobjectT> Subobjects,
         ArgsT &&...Args) {
    if (not TheKey.Definition.has_value()) {
      TheKey.Definition.emplace(Allocator,
                                Name,
                                Subobjects,
                                std::forward<ArgsT>(Args)...);
      return mlir::success();
    }

    if (Name != TheKey.Definition->Name)
      return mlir::failure();

    if (not std::equal(Subobjects.begin(),
                       Subobjects.end(),
                       TheKey.Definition->Subobjects.begin(),
                       TheKey.Definition->Subobjects.end()))
      return mlir::failure();

    return mlir::success();
  }

  [[nodiscard]] uint64_t getID() const { return TheKey.ID; }

  [[nodiscard]] bool isInitialized() const {
    return TheKey.Definition.has_value();
  }

  [[nodiscard]] llvm::StringRef getName() const {
    return TheKey.Definition ? TheKey.Definition->Name : llvm::StringRef{};
  }

  [[nodiscard]] llvm::ArrayRef<SubobjectT> getSubobjects() const {
    return TheKey.Definition ? TheKey.Definition->Subobjects :
                               llvm::ArrayRef<SubobjectT>{};
  }

protected:
  [[nodiscard]] const ValueT &getValue() const { return *TheKey.Definition; }
};

struct StructTypeStorageValue {
  uint64_t Size;
  StructTypeStorageValue(const uint64_t Size) : Size(Size) {}
};

struct StructTypeAttrStorage : ClassTypeStorage<StructTypeAttrStorage,
                                                mlir::AttributeStorage,
                                                FieldAttr,
                                                StructTypeStorageValue> {
  using ClassTypeStorage::ClassTypeStorage;

  [[nodiscard]] uint64_t getSize() const { return getValue().Size; }
};

struct UnionTypeAttrStorage
  : ClassTypeStorage<UnionTypeAttrStorage, mlir::AttributeStorage, FieldAttr> {
  using ClassTypeStorage::ClassTypeStorage;
};

struct ScalarTupleTypeStorage : ClassTypeStorage<ScalarTupleTypeStorage,
                                                 mlir::TypeStorage,
                                                 ScalarTupleElementAttr> {
  using ClassTypeStorage::ClassTypeStorage;
};

} // namespace mlir::clift
