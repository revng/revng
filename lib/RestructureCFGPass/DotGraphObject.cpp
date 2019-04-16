/// \file DotGraphObject.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>
#include <map>
#include <set>

// revng includes
#include "revng/Support/Debug.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/DotGraphObject.h"

using namespace llvm;

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
}

template<typename StreamT>
void DotGraph::parseDot(StreamT &S) {

  // Keep a map between the node identifiers of the nodes defined in the dot
  // file and the corresponding `DotNode` object.
  std::map<std::string, DotNode *> DotNodeMap;

  // Keep a counter for the lines, so that we know how to treat the first and
  // last line.
  std::string CurrentLine;
  unsigned LineCounter = 0;

  // Delimiter of source and target nodes of an edge.
  std::string Delimiter = "->";
  size_t DelimiterSize = Delimiter.size();

  // Compute the number of lines present in the input file.
  unsigned LastLine = std::count(std::istreambuf_iterator<char>(S),
                                 std::istreambuf_iterator<char>(),
                                 '\n');

  // Realign the position to the beginning of the file.
  S.clear();
  S.seekg(0, std::ios::beg);

  while (getline(S, CurrentLine)) {
    LineCounter++;

    // Trim away all the spaces from the current line.
    CurrentLine.erase(remove_if(CurrentLine.begin(),
                                CurrentLine.end(),
                                isspace),
                      CurrentLine.end());

    if (LineCounter == 1) {

      // First line should contain the header of the graph.
      revng_assert(CurrentLine == "digraphTestGraph{");
    } else if (LineCounter == LastLine) {

      // Last line should only contain a closing `}`.
      revng_assert(CurrentLine == "}");
    } else {

      // The body of the dot file should contain the edge declaration.
      dbg << "Parsed line:\n";
      dbg << CurrentLine << "\n";
      size_t pos = CurrentLine.find(Delimiter);
      revng_assert(pos != std::string::npos);

      dbg << "Source is: ";
      // Collect the source node (everything that comes before `->`).
      std::string SourceID = CurrentLine.substr(0, pos);
      dbg << SourceID << "\n";

      dbg << "Target is: ";
      // Collect the target node.
      std::string TargetID = CurrentLine.substr(pos+DelimiterSize);

      // Remove the last `;`.
      size_t TargetSize = TargetID.size();
      revng_assert(TargetID.substr(TargetSize-1, TargetSize) == ";");
      TargetID = TargetID.substr(0, TargetSize-1);
      dbg << TargetID << "\n";

      DotNode *Source = nullptr;
      DotNode *Target = nullptr;

      // Lazily creates the nodes that have not been created before and retrieve
      // all the necessary pointers to the nodes.
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
    }
  }

  std::set<DotNode *> EntryCandidates;
  EntryCandidates.insert(nodes().begin(), nodes().end());
  for (DotNode *N: nodes())
    for (DotNode *Succ : N->successors())
      EntryCandidates.erase(Succ);

  // Check that we have a single entry candidate and enforce it.
  revng_assert(EntryCandidates.size() == 1);
  EntryNode = *EntryCandidates.begin();
}

void DotGraph::parseDotFromFile(std::string FileName) {
  std::ifstream DotFile;
  DotFile.open(FileName);

  // Check that the file has been opened.
  if (DotFile.is_open()) {
    parseDot(DotFile);

    // Close the input file.
    DotFile.close();
  } else {
    revng_abort("Input dot file not opened");
  }
}

DotNode *DotGraph::addNode(std::string &Name) {
  Nodes.emplace_back(std::make_unique<DotNode>(Name));
  DotNode *NewNode = Nodes.back().get();
  return NewNode;
}
