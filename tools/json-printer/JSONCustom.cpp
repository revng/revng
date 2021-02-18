/// \file JSONCustom.cpp
/// \brief Print function boundaries information in JSON.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "boost/icl/interval_set.hpp"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

#include "JSONCustom.h"

using namespace llvm;

char JsonCustom::ID = 0;
static RegisterPass<JsonCustom> X("jsoncustom", "Print JSON", true, true);

class CustomJsonWriter {
public:
  CustomJsonWriter(Module &M, raw_fd_ostream &File) :
    CurrentModule(M), File(File) {}

  void run();

private:
  Module &CurrentModule;
  raw_fd_ostream &File;
};

void CustomJsonWriter::run() {

  // Register the range of addresses covered by each basic block
  using interval_set = boost::icl::interval_set<uint64_t>;
  using interval = boost::icl::interval<uint64_t>;
  std::map<BasicBlock *, interval_set> Coverage;
  for (User *U : CurrentModule.getFunction("newpc")->users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (Call == nullptr)
      continue;

    BasicBlock *BB = Call->getParent();
    auto MA = MetaAddress::fromConstant(Call->getOperand(0));
    uint64_t Address = MA.address();
    uint64_t Size = getLimitedValue(Call->getOperand(1));
    revng_assert(Address > 0 && Size > 0);

    Coverage[BB] += interval::right_open(Address, Address + Size);
  }

  const char *FunctionDelimiter = "";

  // Header
  File << "[";

  for (Function &F : CurrentModule) {

    // Analyze only isolated functions
    if (F.getName().startswith("bb.")) {

      // Function header
      File << FunctionDelimiter << "\n  {\n";

      // Function entry
      BasicBlock &DummyEntry = F.getEntryBlock();
      BasicBlock *Entry = DummyEntry.getSingleSuccessor();
      File << "    \"entry_point\": \"";
      File << Entry->getName();
      File << "\",\n";
      File << "    \"entry_point_address\": \"";
      std::stringstream EntryAddress;
      EntryAddress << "0x" << std::hex << getBasicBlockPC(Entry).address();
      File << EntryAddress.str();
      File << "\",\n";

      interval_set FunctionCoverage;

      // Basic blocks
      const char *BasicBlockDelimiter = "";
      File << "    \"basic_blocks\": [";
      for (BasicBlock &BB : F) {
        File << BasicBlockDelimiter;
        File << "{\"name\": \"" << BB.getName() << "\", ";
        auto It = Coverage.find(&BB);
        if (It != Coverage.end()) {
          const interval_set &IntervalSet = It->second;
          FunctionCoverage += IntervalSet;
          revng_assert(IntervalSet.iterative_size() == 1);
          const auto &Range = *(IntervalSet.begin());
          std::stringstream Addresses;
          Addresses << "\"start\": \"0x" << std::hex << Range.lower() << "\", ";
          Addresses << "\"end\": \"0x" << std::hex << Range.upper() << "\"";
          File << Addresses.str();
        } else {
          File << "\"start\": \"\", \"end\": \"\"";
        }
        File << "}";
        BasicBlockDelimiter = ", ";
      }
      File << "],\n";

      // Coverage
      const char *CoverageDelimiter = "";
      File << "    \"coverage\": [";
      for (const auto &Range : FunctionCoverage) {
        File << CoverageDelimiter;
        File << "{";
        std::stringstream Addresses;
        Addresses << "\"start\": \"0x" << std::hex << Range.lower() << "\", ";
        Addresses << "\"end\": \"0x" << std::hex << Range.upper() << "\"";
        File << Addresses.str();
        File << "}";
        CoverageDelimiter = ", ";
      }
      File << "]\n";

      // Function footer
      File << "  }";
      FunctionDelimiter = ",";
    }
  }

  // Footer
  File << "\n]\n";
}

bool JsonCustom::runOnModule(Module &M) {
  const std::string Filename = (M.getName() + ".json.revng").str();
  errs() << "Writing '" << Filename << "'...";

  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::F_Text);
  revng_check(not EC, "Error opening file for dumping JSON\n");

  CustomJsonWriter Writer(M, File);
  Writer.run();

  errs() << "done.\n";

  return false;
}
