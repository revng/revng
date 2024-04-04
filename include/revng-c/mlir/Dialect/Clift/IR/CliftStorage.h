#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/StorageUniquer.h"

namespace mlir::clift {

struct StructTypeStorage : public mlir::AttributeStorage {
public:
  // Some internal mechanism of mlir have access to the key of the storage
  // instead of the entire storage or the key using that storage. For that
  // reason it is best to throw everything related to the type inside the key,
  // and then override operators to pretend they don't know about the non key
  // fields. That way when it is needed one can access everything.
  struct Key {
    uint64_t ID;
    llvm::StringRef name;
    uint64_t Size;
    Optional<llvm::SmallVector<FieldAttr, 2>> fields;

    // struct storages are never exposed to the user, they are only used
    // internally to figure out how to create unique objects. only operator== is
    // every used, everything else is handled by hasValue
    bool operator==(const Key &Other) const { return Other.ID == ID; }

    llvm::hash_code hashValue() const { return llvm::hash_value(ID); }

    [[nodiscard]] bool isInitialized() const { return fields.has_value(); }
  };

  using KeyTy = Key;

  static llvm::hash_code hashKey(const KeyTy &key) { return key.hashValue(); }

  /// Construct the storage from the type name. Explicitly initialize the
  /// containedType to nullptr, which is used as marker for the mutable
  /// component being not yet initialized.
  StructTypeStorage(uint64_t ID) { TheKey.ID = ID; }

  bool operator==(const Key &Other) const { return this->TheKey == Other; }

  /// Define a construction method for creating a new instance of the storage.
  static StructTypeStorage *
  construct(mlir::StorageUniquer::StorageAllocator &allocator,
            const KeyTy &Key) {
    auto ToReturn = new (allocator.allocate<StructTypeStorage>())
      StructTypeStorage(Key.ID);
    if (Key.isInitialized()) {
      auto Res = ToReturn->mutate(allocator, Key.name, Key.Size, *Key.fields);
      revng_assert(Res.succeeded());
    }

    return ToReturn;
  }

  /// Define a mutation method for changing the type after it is created. In
  /// many cases, we only want to set the mutable component once and reject
  /// any further modification, which can be achieved by returning failure
  /// from this function.
  mlir::LogicalResult mutate(mlir::StorageUniquer::StorageAllocator &alloc,
                             llvm::StringRef name,
                             uint64_t Size,
                             llvm::ArrayRef<FieldAttr> body) {
    if (TheKey.fields.has_value() and body == *TheKey.fields
        and TheKey.name == name and TheKey.Size == Size)
      return mlir::success();

    if (TheKey.fields.has_value())
      return mlir::failure();

    TheKey.fields = llvm::SmallVector<FieldAttr, 2>();
    for (auto field : body)
      TheKey.fields->push_back(field);
    TheKey.name = alloc.copyInto(name);
    TheKey.Size = Size;
    return mlir::success();
  }

  [[nodiscard]] llvm::StringRef getName() const { return TheKey.name; }

  [[nodiscard]] bool isInitialized() const { return TheKey.isInitialized(); }

  llvm::ArrayRef<FieldAttr> getFields() const {
    revng_assert(isInitialized());
    return *TheKey.fields;
  }

  uint64_t getSize() const { return TheKey.Size; }

  uint64_t getID() const { return TheKey.ID; }

private:
  Key TheKey;
};

struct UnionTypeStorage : public mlir::AttributeStorage {
public:
  struct Key {

    uint64_t ID;
    llvm::StringRef name;
    Optional<llvm::SmallVector<FieldAttr, 2>> fields;

    bool operator==(const Key &Other) const { return Other.ID == ID; }

    llvm::hash_code hashValue() const { return llvm::hash_value(ID); }

    [[nodiscard]] bool isInitialized() const { return fields.has_value(); }
  };

  using KeyTy = Key;

  static llvm::hash_code hashKey(const KeyTy &key) { return key.hashValue(); }

  UnionTypeStorage(uint64_t ID) { TheKey.ID = ID; }

  /// Define the comparison function.
  bool operator==(const KeyTy &key) const { return key == TheKey; }

  /// Define a construction method for creating a new instance of the storage.
  static UnionTypeStorage *
  construct(mlir::StorageUniquer::StorageAllocator &allocator,
            const KeyTy &Key) {
    auto ToReturn = new (allocator.allocate<UnionTypeStorage>())
      UnionTypeStorage(Key.ID);
    if (Key.isInitialized()) {
      auto Res = ToReturn->mutate(allocator, Key.name, *Key.fields);
      revng_assert(Res.succeeded());
    }

    return ToReturn;
  }

  /// Define a mutation method for changing the type after it is created. In
  /// many cases, we only want to set the mutable component once and reject
  /// any further modification, which can be achieved by returning failure
  /// from this function.
  mlir::LogicalResult mutate(mlir::StorageUniquer::StorageAllocator &alloc,
                             llvm::StringRef name,
                             llvm::ArrayRef<FieldAttr> body) {
    if (TheKey.fields.has_value() and body == *TheKey.fields
        and TheKey.name == name)
      return mlir::success();

    if (TheKey.fields.has_value())
      return mlir::failure();

    TheKey.fields = llvm::SmallVector<FieldAttr, 2>();
    for (auto field : body)
      TheKey.fields->push_back(field);
    TheKey.name = alloc.copyInto(name);
    return mlir::success();
  }

  [[nodiscard]] llvm::StringRef getName() const { return TheKey.name; }

  [[nodiscard]] bool isInitialized() const { return TheKey.isInitialized(); }

  llvm::ArrayRef<FieldAttr> getFields() const {

    revng_assert(isInitialized());
    return *TheKey.fields;
  }

  uint64_t getID() const { return TheKey.ID; }

private:
  Key TheKey;
};
} // namespace mlir::clift
