#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <variant>

#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Concepts.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreePath.h"
#include "revng/TupleTree/Visits.h"

template<typename T, typename RootType>
class TupleTreeReference {
public:
  using RootT = RootType;
  using RootVariant = std::variant<RootT *, const RootT *>;

public:
  RootVariant Root = static_cast<RootT *>(nullptr);
  TupleTreePath Path;

public:
  static TupleTreeReference
  fromPath(ConstOrNot<TupleTreeReference::RootT> auto *Root,
           const TupleTreePath &Path) {
    return TupleTreeReference{ .Root = RootVariant{ Root }, .Path = Path };
  }

  static TupleTreeReference
  fromString(ConstOrNot<TupleTreeReference::RootT> auto *Root,
             llvm::StringRef Path) {
    std::optional<TupleTreePath> OptionalPath = stringAsPath<RootT>(Path);
    if (not OptionalPath.has_value())
      return TupleTreeReference{};
    return fromPath(Root, *OptionalPath);
  }

  bool operator==(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    return Path == Other.Path;
  }

  bool operator<(const TupleTreeReference &Other) const {
    // The paths are the same even if they are referred to different roots
    return Path < Other.Path;
  }

public:
  std::string toString() const { return *pathAsString<RootT>(Path); }

  const TupleTreePath &path() const { return Path; }

private:
  bool hasNullRoot() const {
    const auto IsNullVisitor = [](const auto &Pointer) {
      return Pointer == nullptr;
    };
    return std::visit(IsNullVisitor, Root);
  }

  bool canGet() const {
    return Root.index() != std::variant_npos and not hasNullRoot();
  }

public:
  const T *get() const {
    revng_assert(canGet());

    if (Path.size() == 0)
      return nullptr;

    const auto GetByPathVisitor = [&Path = Path](const auto &RootPointer) {
      return getByPath<T>(Path, *RootPointer);
    };

    return std::visit(GetByPathVisitor, Root);
  }

  T *get() {
    revng_assert(canGet());

    if (Path.size() == 0)
      return nullptr;

    if (std::holds_alternative<RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<RootT *>(Root));
    } else if (std::holds_alternative<const RootT *>(Root)) {
      revng_abort("Called get() with const root, use getConst!");
    } else {
      revng_abort("Invalid root variant!");
    }
  }

  const T *getConst() const {
    revng_assert(canGet());

    if (Path.size() == 0)
      return nullptr;

    if (std::holds_alternative<const RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<const RootT *>(Root));
    } else if (std::holds_alternative<RootT *>(Root)) {
      return getByPath<T>(Path, *std::get<RootT *>(Root));
    } else {
      revng_abort("Invalid root variant!");
    }
  }

  bool isValid() const debug_function {
    return canGet() and not Path.empty() and get() != nullptr;
  }
};

template<typename T>
concept IsTupleTreeReference = is_specialization_v<T, TupleTreeReference>;

template<IsTupleTreeReference T>
struct llvm::yaml::ScalarTraits<T> {

  static void output(const T &Obj, void *, llvm::raw_ostream &Out) {
    Out << Obj.toString();
  }

  static llvm::StringRef input(llvm::StringRef Path, void *, T &Obj) {
    // We temporarily initialize Root to nullptr, a post-processing phase will
    // take care of fixup these
    Obj = T::fromString(static_cast<typename T::RootT *>(nullptr), Path);
    return {};
  }

  static auto mustQuote(llvm::StringRef) {
    return llvm::yaml::QuotingType::Double;
  }
};

/// \brief Specialization for the std::variant we have in TupleTreeReference
template<bool X, typename T>
inline void writeToLog(Logger<X> &This,
                       const std::variant<T *, const T *> &Var,
                       int Ignored) {
  if (Var.index() == std::variant_npos)
    writeToLog(This, llvm::StringRef("std::variant_npos"), Ignored);
  else if (std::holds_alternative<T *>(Var))
    writeToLog(This, std::get<T *>(Var), Ignored);
  else if (std::holds_alternative<const T *>(Var))
    writeToLog(This, std::get<const T *>(Var), Ignored);
}
