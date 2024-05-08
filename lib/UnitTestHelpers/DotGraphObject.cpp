/// \file DotGraphObject.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <fstream>
#include <map>
#include <set>

#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/UnitTestHelpers/DotGraphObject.h"

using namespace llvm;

Logger<> DotLogger("parsedot");

void DotNode::addSuccessor(DotNode *NewSuccessor) {

  // Assert that we are not double inserting.
  bool Found = false;
  for (DotNode *Successor : Successors) {
    if (NewSuccessor == Successor) {
      Found = true;
      break;
    }
  }
  revng_assert(not Found);

  Successors.push_back(NewSuccessor);
  SuccEdges.push_back(std::make_pair(this, NewSuccessor));
}

void DotNode::addPredecessor(DotNode *NewPredecessor) {

  // Assert that we are not double inserting.
  bool Found = false;
  for (DotNode *Predecessor : Predecessors) {
    if (NewPredecessor == Predecessor) {
      Found = true;
      break;
    }
  }
  revng_assert(not Found);

  Predecessors.push_back(NewPredecessor);
  PredEdges.push_back(std::make_pair(NewPredecessor, this));
}

void DotNode::printAsOperand(llvm::raw_ostream &O, bool /* PrintType */) const {
  O << Name;
}

void DotGraph::parseDotImpl(std::ifstream &F, llvm::StringRef EntryName) {

  // Keep a map between the node identifiers of the nodes defined in the dot
  // file and the corresponding `DotNode` object.
  std::map<llvm::SmallString<8>, DotNode *> DotNodeMap;

  // Container for the current line.
  std::string CurrentLine;

  // Edge delimiter.
  llvm::StringRef Delimiter("->");
  size_t DelimiterSize = Delimiter.size();

  bool ParsedHeader = false;
  bool ParsedFooter = false;

  while (getline(F, CurrentLine)) {

    // Drop comments, i.e., everything after #
    size_t SharpPosition = CurrentLine.find("#");
    if (SharpPosition != std::string::npos) {
      CurrentLine.resize(SharpPosition);
    }

    // Trim away all the spaces from the current line.
    CurrentLine.erase(remove_if(CurrentLine.begin(),
                                CurrentLine.end(),
                                isspace),
                      CurrentLine.end());

    // Obtain a StringRef which is more easily to handle.
    llvm::StringRef CurrentLineRef = CurrentLine;

    // Always skip over empty lines.
    if (CurrentLineRef == "") {
      continue;
    } else {
      if (not ParsedHeader) {
        if (CurrentLineRef.startswith("digraph")
            and CurrentLineRef.endswith("{")) {
          ParsedHeader = true;
          continue;
        } else {
          revng_abort("GraphViz header malformed");
        }
      } else if (not ParsedFooter) {

        if (CurrentLineRef == "}") {
          ParsedFooter = true;
          continue;
        }

        // The body of the dot file should contain the edge declaration.
        revng_log(DotLogger, "Parsed line:\n" << CurrentLineRef << "\n");
        size_t Pos = CurrentLineRef.find(Delimiter);
        revng_assert(Pos != llvm::StringRef::npos);

        // Collect the source node (everything that comes before `->`).
        llvm::StringRef SourceID = CurrentLineRef.substr(0, Pos);
        revng_log(DotLogger, "Source is: " << SourceID << "\n");

        // Collect the target node.
        llvm::StringRef TargetID = CurrentLineRef.substr(Pos + DelimiterSize);

        // Remove the last `;`.
        size_t TargetSize = TargetID.size();
        revng_assert(TargetID.substr(TargetSize - 1, TargetSize) == ";");
        TargetID = TargetID.substr(0, TargetSize - 1);
        revng_log(DotLogger, "Target is: " << TargetID << "\n");

        DotNode *Source = nullptr;
        DotNode *Target = nullptr;

        // Lazily creates the nodes that have not been created before and
        // retrieve all the necessary pointers to the nodes.
        auto It = DotNodeMap.find(SourceID);
        if (It == DotNodeMap.end()) {
          Source = addNode(SourceID);
          DotNodeMap[SourceID] = Source;
        } else {
          Source = It->second;
        }

        It = DotNodeMap.find(TargetID);
        if (It == DotNodeMap.end()) {
          Target = addNode(TargetID);
          DotNodeMap[TargetID] = Target;
        } else {
          Target = It->second;
        }

        revng_assert(Source != nullptr and Target != nullptr);

        // Add to the successors of `Source` the node `Target`.
        Source->addSuccessor(Target);
        Target->addPredecessor(Source);
      } else {
        revng_abort("Content after the footer.");
      }
    }
  }

  // Set the entry node searching for the entry name.
  if (not EntryName.empty()) {
    EntryNode = getNodeByName(EntryName);
  } else {
    EntryNode = nullptr;
  }
}

void DotGraph::parseDotFromFile(llvm::StringRef FileName,
                                llvm::StringRef EntryName) {
  std::ifstream DotFile;
  DotFile.open(FileName.data());

  // Check that the file has been opened.
  if (DotFile.is_open()) {
    parseDotImpl(DotFile, EntryName);

    // Close the input file.
    DotFile.close();
  } else {
    revng_abort("Input dot file not opened");
  }
}

DotNode *DotGraph::addNode(llvm::StringRef Name) {
  Nodes.emplace_back(std::make_unique<DotNode>(Name, this));
  DotNode *NewNode = Nodes.back().get();
  return NewNode;
}

DotNode *DotGraph::getNodeByName(llvm::StringRef Name) {

  // Verify that we have a single node matching the name.
  DotNode *Candidate = nullptr;

  for (DotNode *Node : nodes()) {
    if (Node->getName() == Name) {
      if (Candidate == nullptr) {
        Candidate = Node;
      } else {
        revng_abort("Two nodes with the same name found.");
      }
    }
  }

  // Check that we found the node we are searching for.
  revng_assert(Candidate != nullptr);
  return Candidate;
}
