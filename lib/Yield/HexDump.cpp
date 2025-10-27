/// \file HexDump.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "boost/icl/interval_map.hpp"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress/IntervalContainers.h"
#include "revng/Yield/HexDump.h"

using namespace llvm;

namespace revng::pipes {

static const size_t BytesInLine = 16;

static FormattedNumber formatNumber(uint64_t Number, unsigned Width = 8) {
  return FormattedNumber(Number, 0, Width, true, false, false);
};

using CFG = efa::ControlFlowGraph;
using CFGGetter = std::function<const CFG &(const MetaAddress &)>;

static void outputHexDump(const model::Binary &Binary,
                          llvm::ArrayRef<const llvm::Function *> Functions,
                          CFGGetter CFGGetter,
                          llvm::StringRef BinaryBuffer,
                          llvm::raw_ostream &Output) {
  RawBinaryView BinaryView(Binary, BinaryBuffer);

  using boost::icl::discrete_interval;
  using boost::icl::inplace_plus;
  using boost::icl::inter_section;
  using boost::icl::interval_map;
  using boost::icl::partial_absorber;
  using IntervalType = discrete_interval<IntervalMetaAddress>;
  using Map = interval_map<IntervalMetaAddress,
                           std::set<std::string>,
                           partial_absorber,
                           std::less,
                           inplace_plus,
                           inter_section,
                           IntervalType>;
  Map Instructions;
  ptml::MarkupBuilder B;

  auto CreateTag = [&B](const std::string &Location) -> ptml::Tag {
    auto Tag = B.getTag("span");
    Tag.addAttribute("data-location-definition", Location);
    return Tag;
  };

  for (const Function *F : Functions) {
    MetaAddress Address = getMetaAddressOfIsolatedFunction(*F);
    const efa::ControlFlowGraph &Metadata = CFGGetter(Address);
    MetaAddress EntryAddress = Metadata.Entry();

    for (const Instruction &I : llvm::instructions(F)) {
      if (auto *Call = getCallTo(&I, "newpc")) {
        const BasicBlock *JumpTarget = getJumpTargetBlock(I.getParent());
        revng_assert(JumpTarget != nullptr);
        auto BasicBlockID = blockIDFromNewPC(JumpTarget->getFirstNonPHI());

        MetaAddress Address = MetaAddress::fromValue(Call->getArgOperand(0));

        auto *SizeValue = dyn_cast<ConstantInt>(Call->getArgOperand(1));

        uint64_t Size = SizeValue->getZExtValue();
        MetaAddress Begin = Address.toGeneric();
        MetaAddress End = Begin + Size;
        auto Interval = IntervalType::right_open(Begin, End);

        std::string Str = locationString(ranks::Instruction,
                                         EntryAddress,
                                         BasicBlockID,
                                         Address);

        std::set<std::string> Set{ Str };
        Instructions.add(std::make_pair(Interval, Set));
      }
    }
  }

  ptml::Tag DivTag = B.getTag("div");

  Output << DivTag.open();

  MetaAddress CurrentAddress;
  Map::const_iterator Current = Instructions.begin();
  Map::const_iterator Next = std::next(Current);
  const Map::const_iterator End = Instructions.end();

  std::stack<ptml::Tag> OpenedTags;
  for (const auto &[Segment, SegmentBinary] : BinaryView.segments()) {
    MetaAddress CurrentAddress = Segment.StartAddress();
    size_t Counter = 0;

    // Stores bytes from current line as printable characters
    SmallString<16> PrintableChars;

    for (size_t Index = 0; Index < SegmentBinary.size(); ++Index) {
      // If there is still some interval to process, set it to CurrentInterval.
      // Otherwise, create invalid interval.
      IntervalType CurrentInterval = (Current != End) ?
                                       Current->first :
                                       IntervalType{ IntervalMetaAddress{},
                                                     IntervalMetaAddress{} };

      bool LineBegins = Index % BytesInLine == 0;
      if (LineBegins) {
        // Print address of first byte in line
        Output << formatNumber(CurrentAddress.address()) << "  ";
      }

      // Open tags if this is beginning of next interval or line begins
      bool IsIntervalValid = CurrentInterval.lower().isValid()
                             and CurrentInterval.upper().isValid();
      // Check if current byte is inside currently processed interval
      bool AfterStart = CurrentInterval.lower() <= CurrentAddress;
      // Check if current byte is before end of current interval
      bool BeforeEnd = CurrentAddress < CurrentInterval.upper();
      // If current interval is valid and current byte is after start and before
      // end of interval, this byte belongs to some interval and should be
      // wrapped with location tags.
      bool IsInsideInterval = IsIntervalValid and AfterStart and BeforeEnd;
      // Is current byte the first byte of interval?
      bool AtStart = CurrentInterval.lower() == CurrentAddress;
      // If interval is valid and current byte is the first byte, location tag
      // should be printed before byte.
      bool IsStartOfInterval = IsIntervalValid and AtStart;

      // Tag opening is printed in two situations:
      //  1. new interval is beginning on current byte
      //  2. new line begins and previously opened (and closed on line end)
      //     interval is continued.
      if (IsStartOfInterval or (LineBegins and IsInsideInterval)) {
        for (const std::string &Tag : Current->second) {
          auto PTMLTag = CreateTag(Tag);
          OpenedTags.push(PTMLTag);
          Output << PTMLTag.open();
        }
      }

      // Format number and put it to the output
      uint64_t B = SegmentBinary[Index];
      Output << formatNumber(B, 2);

      // Increment counter of bytes printed in current line.
      ++Counter;

      // Append printable character.
      if (std::isprint(B)) {
        PrintableChars += B;
      } else {
        PrintableChars += '.';
      }

      MetaAddress NextAddress = CurrentAddress + 1;

      const bool EndOfLine = Counter == BytesInLine;
      // If current byte (just printed) is in current interval, but next byte
      // isn't, this place is end of interval.
      const bool EndOfInterval = CurrentAddress < CurrentInterval.upper()
                                 and NextAddress >= CurrentInterval.upper();

      const bool EndOfSegment = Index + 1 == SegmentBinary.size();

      // All opened tags has to be closed now if current byte is still inside
      // some interval and:
      //  1. next byte is not in current interval (end of interval) OR
      //  2. after just printed by there is end of line
      if (IsInsideInterval and (EndOfInterval or EndOfLine)) {
        while (not OpenedTags.empty()) {
          auto &PTMLTag = OpenedTags.top();
          Output << PTMLTag.close();
          OpenedTags.pop();
        }
      }

      // At the end of each printed line of bytes in hex format (with location
      // tags), ASCII representation of current line is printed.
      if (EndOfLine) {
        // Output ASCII representation at the end of the line
        std::string Temp;
        raw_string_ostream Printable(Temp);
        printHTMLEscaped(PrintableChars, Printable);
        Output << "   | " << Printable.str() << " |\n";
        PrintableChars.clear();

        // At the end of line, Counter is set to 0.
        Counter = 0;
      } else {
        // Put space separating consecutive bytes
        Output << ' ';

        if (Counter == 8) {
          // After every 8 bytes, put extra space
          Output << ' ';
        }
      }

      // At the end of interval, we try to go to the next interval (if it
      // exists).
      if (EndOfInterval) {
        Current = Next;
        if (Next != End) {
          Next = std::next(Next);
        }
      }

      CurrentAddress = NextAddress;
    }

    // After each binary segment, we add extra empty line.
    Output << '\n';
  }

  Output << DivTag.close();
}

class HexDumpPipe {
public:
  static constexpr auto Name = "hex-dump";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    return { ContractGroup({ Contract(kinds::Binary,
                                      0,
                                      kinds::HexDump,
                                      3,
                                      InputPreservation::Preserve),
                             Contract(kinds::Isolated,
                                      1,
                                      kinds::HexDump,
                                      3,
                                      InputPreservation::Preserve),
                             Contract(kinds::CFG,
                                      2,
                                      kinds::HexDump,
                                      3,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &SourceBinary,
           const pipeline::LLVMContainer &ModuleContainer,
           const CFGMap &CFGMap,
           HexDumpFileContainer &Output) {

    // This pipe works only if we have all the targets
    pipeline::TargetsList FunctionList = ModuleContainer.enumerate();
    if (not FunctionList.contains(kinds::Isolated.allTargets(EC.getContext())))
      return;

    if (not SourceBinary.exists())
      return;

    pipeline::TargetsList CFGList = CFGMap.enumerate();
    if (not CFGList.contains(kinds::CFG.allTargets(EC.getContext())))
      return;

    const model::Binary &Binary = *getModelFromContext(EC);

    std::vector<const llvm::Function *> Functions;
    for (const llvm::Function &F :
         FunctionTags::Isolated.functions(&ModuleContainer.getModule())) {
      Functions.push_back(&F);
    }

    ControlFlowGraphCache CFGCache(CFGMap);
    auto CFGGetter =
      [&CFGCache](const MetaAddress &Address) -> const efa::ControlFlowGraph & {
      return CFGCache.getControlFlowGraph(Address);
    };

    auto Buffer = revng::cantFail(MemoryBuffer::getFile(*SourceBinary.path()));

    std::error_code ErrorCode;
    raw_fd_ostream OutputOS(Output.getOrCreatePath(),
                            ErrorCode,
                            sys::fs::CD_CreateAlways);

    revng_assert(not ErrorCode, "Could not open file!");

    // Proceed with emission
    outputHexDump(Binary, Functions, CFGGetter, Buffer->getBuffer(), OutputOS);

    EC.commitUniqueTarget(Output);
  }
};

} // namespace revng::pipes
  //
static pipeline::RegisterPipe<revng::pipes::HexDumpPipe> X;

namespace revng::pypeline::piperuns {

void HexDump::run(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  const BinariesContainer &BinaryContainer,
                  const LLVMFunctionContainer &ModuleContainer,
                  const CFGMap &CFG,
                  HexDumpContainer &Output) {

  const model::Binary &Binary = *Model.get().get();

  auto Buffer = BinaryContainer.getFile(0);
  auto OutputOS = Output.getOStream(ObjectID{});

  std::vector<const llvm::Function *> Functions;
  for (const model::Function &Function : Binary.Functions()) {
    const llvm::Module &Module = ModuleContainer
                                   .getModule(ObjectID(Function.Entry()));
    for (const llvm::Function &LLVMFunction :
         FunctionTags::Isolated.functions(&Module)) {
      Functions.push_back(&LLVMFunction);
    }
  }

  auto CFGGetter =
    [&CFG](const MetaAddress &Address) -> const efa::ControlFlowGraph & {
    return *CFG.getElement(ObjectID(Address));
  };

  ::revng::pipes::outputHexDump(Binary,
                                Functions,
                                CFGGetter,
                                { Buffer.data(), Buffer.size() },
                                *OutputOS);
}

} // namespace revng::pypeline::piperuns
