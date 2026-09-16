//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

#include "revng/Model/RawBinaryView.h"
#include "revng/Pipebox/Helpers.h"
#include "revng/SegmentReferences/DetectCStrings.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/Unicode.h"

using namespace llvm;

static Logger Log("detect-c-strings");

using Encoding = UnicodeCStringView::Encoding;
using Candidate = DetectCStrings::Candidate;
using Credibility = DetectCStrings::Credibility;

/// A string this long says it is a string all by itself.
static constexpr size_t ConvincingCodePointCount = 5;

/// How many strings packed back to back it takes for them to vouch for each
/// other, however short they are.
static constexpr size_t ConvincingRunLength = 2;

static StringRef encodingName(Encoding TheEncoding) {
  switch (TheEncoding) {
  case Encoding::Invalid:
    return "invalid";
  case Encoding::UTF8:
    return "UTF-8";
  case Encoding::UTF16LE:
    return "UTF-16LE";
  case Encoding::UTF16BE:
    return "UTF-16BE";
  }

  revng_abort("Unknown encoding");
}

static void logCandidate(StringRef Prefix, const Candidate &TheCandidate) {
  if (not Log.isEnabled())
    return;

  const UnicodeCStringView &String = TheCandidate.String;
  Log << Prefix << " \"";
  printEscapedString(String.data(), *Log.getAsLLVMStream());
  Log << "\" (" << String.codePointCount() << " code points, "
      << encodingName(String.encoding()) << ") at "
      << TheCandidate.Address.toString();
  Log << DoLog;
}

void DetectCStrings::run(llvm::Module &M, llvm::Function *LimitTo) {
  revng_log(Log, "Collecting candidates");
  LoggerIndent Indent(Log);

  for (auto &&SegmentUse : SegmentUses.getUses(M, LimitTo)) {
    revng_log(Log,
              "Considering segment use "
                << getName(SegmentUse.TheUse->getUser()) << ". Address is "
                << SegmentUse.Address.toString() << ".");
    LoggerIndent Indent(Log);
    auto MaybeData = BinaryView.getFromAddressOn(SegmentUse.Address);

    // Check if there's data at the given offset
    if (not MaybeData.has_value()) {
      revng_log(Log, "No data, bailing out");
      continue;
    }

    // Record every string this address could spell. Which one it actually
    // spells, if any, is for `commit` to say.
    for (const UnicodeCStringView &String :
         UnicodeCStringView::getPrintable(*MaybeData)) {
      const Candidate &Added = Candidates.add(SegmentUse.Address, String);
      logCandidate("Collected", Added);
    }
  }
}

void DetectCStrings::CandidateList::sortAndDeduplicate() {
  llvm::sort(Candidates, [](const Candidate &LHS, const Candidate &RHS) {
    if (LHS.Address != RHS.Address)
      return LHS.Address < RHS.Address;
    return LHS.String.encoding() < RHS.String.encoding();
  });

  auto IsDuplicate = [](const Candidate &LHS, const Candidate &RHS) {
    return LHS.Address == RHS.Address
           and LHS.String.encoding() == RHS.String.encoding();
  };
  Candidates.erase(llvm::unique(Candidates, IsDuplicate), Candidates.end());
}

void DetectCStrings::CandidateList::approve(llvm::ArrayRef<Candidate *> Run) {
  if (Run.size() < ConvincingRunLength)
    return;

  for (Candidate *Entry : Run) {
    Entry->Rating = Credibility::InCompany;
    logCandidate("Approved (part of a run)", *Entry);
  }
}

void DetectCStrings::CandidateList::rate() {
  revng_log(Log, "Rating " << Candidates.size() << " candidates");
  LoggerIndent Indent(Log);

  // A short string is credible in company: strings packed back to back, all
  // with the same encoding, are a string table, and every entry of one is
  // vouched for by its neighbors.
  //
  // Each encoding is walked on its own: a run is made of readings in one
  // encoding, and the readings of the same addresses in the other encodings sit
  // in between, so walking them all together would break every run up.
  for (Encoding TheEncoding :
       { Encoding::UTF8, Encoding::UTF16LE, Encoding::UTF16BE }) {
    llvm::SmallVector<Candidate *> Run;

    for (Candidate &Current : Candidates) {
      if (Current.String.encoding() != TheEncoding)
        continue;

      if (not Run.empty() and not Run.back()->isFollowedBy(Current)) {
        approve(Run);
        Run.clear();
      }

      Run.push_back(&Current);
    }

    approve(Run);
  }

  // A long string needs no company, so it has the final say over whatever the
  // company it happens to keep had to offer.
  for (Candidate &Current : Candidates) {
    if (Current.String.codePointCount() > ConvincingCodePointCount) {
      Current.Rating = Credibility::OnItsOwn;
      logCandidate("Approved (long enough)", Current);
    }
  }
}

cppcoro::generator<llvm::ArrayRef<Candidate>>
DetectCStrings::CandidateList::byAddress() const {
  llvm::ArrayRef<Candidate> Remaining = Candidates;

  while (not Remaining.empty()) {
    // An address usually has a single reading, so the end of the group is
    // walked to rather than searched for: a binary search would probe the whole
    // of what is left to find a boundary that is almost always right here.
    size_t Count = 1;
    while (Count < Remaining.size()
           and Remaining[Count].Address == Remaining.front().Address)
      ++Count;

    co_yield Remaining.take_front(Count);
    Remaining = Remaining.drop_front(Count);
  }
}

void DetectCStrings::commit() {
  Candidates.sortAndDeduplicate();
  Candidates.rate();

  revng_log(Log, "Committing");
  LoggerIndent Indent(Log);

  // Keep only on candidate per address
  for (llvm::ArrayRef<Candidate> Readings : Candidates.byAddress()) {
    auto ByRating = [](const Candidate &LHS, const Candidate &RHS) {
      return LHS.Rating < RHS.Rating;
    };
    const Candidate *Winner = std::max_element(Readings.begin(),
                                               Readings.end(),
                                               ByRating);

    if (Winner->Rating == Credibility::None)
      Winner = nullptr;

    if (Log.isEnabled())
      for (const Candidate &Reading : Readings)
        if (&Reading != Winner)
          logCandidate("Discarded", Reading);

    if (Winner == nullptr)
      continue;

    // Create a uint8_t array of the size of the string (including the NUL byte)
    unsigned CharSize = Winner->String.charSize();
    auto Character = model::PrimitiveType::makeConstUnsigned(CharSize);
    size_t Size = Winner->String.data().size() / CharSize;
    auto StringType = model::ArrayType::make(std::move(Character), Size);
    bool Success = GlobalBuilder.insert(Winner->Address, std::move(StringType));

    logCandidate(Success ? "Added" : "Not added", *Winner);
  }
}

namespace revng::pypeline::analyses {

llvm::Error DetectCStrings::run(Model &Model,
                                const Request &Incoming,
                                llvm::StringRef Configuration,
                                const BinariesContainer &Binaries,
                                LLVMFunctionContainer &ModuleContainer) {
  RawBinaryView BinaryView = makeBinaryView(Model, Binaries);
  ::DetectCStrings StringDetector(*Model.get().get(), BinaryView);

  for (const ObjectID *Object : Incoming[1])
    StringDetector.run(ModuleContainer.getModule(*Object));

  StringDetector.commit();

  return llvm::Error::success();
}

} // namespace revng::pypeline::analyses
