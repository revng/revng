#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/GlobalVariableBuilder.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"
#include "revng/Support/Generator.h"
#include "revng/Support/Unicode.h"

class RawBinaryView;

/// Analyze segment references in the code, detect pointers to global strings
/// and add corresponding fields in the struct describing segments.
class DetectCStrings {
public:
  enum class Credibility {
    /// Nothing to say for itself.
    None,
    /// Short, but packed back to back with strings just like it.
    InCompany,
    /// Long enough to say it is a string all by itself.
    OnItsOwn
  };

  struct Candidate {
    MetaAddress Address;
    UnicodeCStringView String;
    Credibility Rating = Credibility::None;

    bool isFollowedBy(const Candidate &Next) const {
      MetaAddress End = Address + String.data().size();
      return End.isValid() and Next.Address == End;
    }
  };

  class CandidateList {
  private:
    std::vector<Candidate> Candidates;

  public:
    /// Take note that \p Address could spell \p String.
    const Candidate &add(MetaAddress Address,
                         const UnicodeCStringView &String) {
      return Candidates.emplace_back(Address, String);
    }

    void sortAndDeduplicate();

    void rate();

    /// \return sets of candidates on the same address
    cppcoro::generator<llvm::ArrayRef<Candidate>> byAddress() const;

  private:
    void approve(llvm::ArrayRef<Candidate *> Run);
  };

private:
  SegmentUsesEnumerator SegmentUses;
  RawBinaryView &BinaryView;
  model::GlobalVariableBuilder GlobalBuilder;
  CandidateList Candidates;

public:
  DetectCStrings(model::Binary &Binary, RawBinaryView &BinaryView) :
    SegmentUses(Binary, SegmentUsesEnumerator::SegmentAccess::ReadOnly),
    BinaryView(BinaryView),
    GlobalBuilder(Binary) {}

public:
  void run(llvm::Module &M, llvm::Function *LimitTo = nullptr);

  void commit();
};

namespace revng::pypeline::analyses {

class DetectCStrings {
public:
  static constexpr llvm::StringRef Name = "detect-c-strings";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const BinariesContainer &Binaries,
                  LLVMFunctionContainer &ModuleContainer);
};

} // namespace revng::pypeline::analyses
