#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/TupleTree/TupleTreePath.h"

namespace model {

class Binary;
}
namespace revng {

class LocationBase {
public:
  virtual std::string toString() const = 0;
  virtual ~LocationBase() = default;
};

template<typename RootType>
class TupleTreeLocation : public LocationBase {
private:
  TupleTreePath Path;

public:
  TupleTreeLocation(TupleTreePath Path) : Path(Path) {}

  std::string toString() const override {
    auto AsString = pathAsString<RootType>(Path);
    revng_assert(AsString.has_value());
    return *AsString;
  }

  ~TupleTreeLocation() override = default;
  static std::string getTypeName() { return "TupleTreeLocation"; }
};

class DocumentErrorBase : public llvm::ErrorInfo<DocumentErrorBase> {
public:
  inline static char ID = '0';

private:
  struct Entry {
  public:
    Entry(llvm::StringRef Reason, std::unique_ptr<LocationBase> Location) :
      Reason(Reason.str()), Location(std::move(Location)) {}

    std::string Reason;
    std::unique_ptr<LocationBase> Location;
  };

  using iterator = llvm::SmallVector<Entry, 4>::iterator;
  using const_iterator = llvm::SmallVector<Entry, 4>::const_iterator;

protected:
  using IDContainer = llvm::SmallVector<const void *, 4>;
  llvm::SmallVector<Entry, 4> Entries;

  IDContainer Ids;

public:
  explicit DocumentErrorBase(const IDContainer &DerivedIds) :
    Entries(), Ids(std::move(DerivedIds)) {
    Ids.push_back(DocumentErrorBase::classID());
  }
  DocumentErrorBase() { Ids.push_back(DocumentErrorBase::classID()); }
  virtual std::string getTypeName() const = 0;
  virtual std::string getLocationTypeName() const = 0;

  // Returns the class ID for this type.
  static const void *classID() { return &ID; }

  iterator begin() { return Entries.begin(); }

  iterator end() { return Entries.end(); }

  [[nodiscard]] const_iterator begin() const { return Entries.begin(); }

  [[nodiscard]] const_iterator end() const { return Entries.end(); }

  size_t size() const { return Entries.size(); }

  const std::string &getMessage(size_t I) const { return Entries[I].Reason; }
  std::string getLocation(size_t I) const {
    return Entries[I].Location->toString();
  }

  virtual bool isA(const void *const ClassID) const {
    return llvm::find(Ids, ClassID) != Ids.end();
  }

public:
  std::error_code convertToErrorCode() const final {
    return llvm::inconvertibleErrorCode();
  }
  void log(llvm::raw_ostream &OS) const final {
    OS << "Diff Errors\n";
    for (auto &Reason : Entries) {
      OS << "\t" << Reason.Reason << "\n";
    }
  }
};

template<typename Derived, typename Location>
class DocumentError : public DocumentErrorBase {
private:
  static DocumentErrorBase::IDContainer getClassIds() {
    DocumentErrorBase::IDContainer Ids({
      llvm::ErrorInfoBase::classID(),
      Derived::classID(),
      DocumentError::classID(),
    });
    return Ids;
  }

public:
  using LocationType = Location;
  inline static char ID = '0';
  DocumentError(const DocumentErrorBase::IDContainer &Ids = getClassIds()) :
    DocumentErrorBase(Ids) {}
  DocumentError(llvm::StringRef Reason,
                const Location &ReasonLocation,
                const DocumentErrorBase::IDContainer &Ids = getClassIds()) :
    DocumentErrorBase(Ids) {
    addReason(Reason, ReasonLocation);
  }

  void addReason(llvm::StringRef NewReason, const Location &ReasonLocation) {
    Entries.emplace_back(NewReason.str(),
                         std::make_unique<Location>(ReasonLocation));
  }

  std::string getLocationTypeName() const override {
    return Location::getTypeName();
  }

public:
  static llvm::Error makeError(std::unique_ptr<Derived> Content) {
    if (Content->size() != 0)
      return llvm::Error(std::move(Content));
    return llvm::Error::success();
  }

public:
  // Returns the class ID for this type.
  static const void *classID() { return &Derived::ID; }
};

} // namespace revng
