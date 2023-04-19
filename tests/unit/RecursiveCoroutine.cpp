//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <chrono>
#include <coroutine>
#include <functional>
#include <iostream>
#include <vector>

// TODO: increase height up to explosion
// TODO: compare performance and peak memory

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"

#include "DepthFirstVisit.h"
#include "SimpleRecursiveCoroutine.h"

size_t MaxDepth = 0;
size_t Iterations = 0;

int main() {

  //
  // Run a simple recursive coroutine
  //
  std::vector<MyState> MyStateRCS;
  MyStateRCS.emplace_back();
  myCoroutine(MyStateRCS, 0);

  //
  // Visit a simple graph
  //
  Graph SimpleGraph = createSimpleGraph();
#ifdef ITERATIVE

  std::vector<Entry> ThisRCS;

  auto *Root = SimpleGraph.root();
  auto &Children = Root->children();

  ThisRCS.emplace_back(Entry{ Root, true, Children.begin(), Children.end() });
  iterativeFindMaxDepth(ThisRCS);

#else

  std::vector<Node *> ThisRCS;
  // findMaxDepth expects the RCS to already contain the state for the first
  // element
  ThisRCS.emplace_back(SimpleGraph.root());
  // run the coroutine
  findMaxDepth(ThisRCS);

#endif

  std::cerr << "MaxDepth: " << MaxDepth << std::endl;
  std::cerr << "Iterations: " << Iterations << std::endl;
  revng_check(MaxDepth == 4);
  revng_check(Iterations == 7);

  //
  // Visit a complex graph
  //
  Graph G = createRandomGraph();

  using namespace std::chrono;
  using us = long long;
  const us Repeat = 1;
  us Average = 0;

  for (size_t I = 0; I < Repeat; I++) {
    Iterations = 0;
    MaxDepth = 0;

    auto Start = high_resolution_clock::now();
#ifdef ITERATIVE

    std::vector<Entry> Stack;

    auto *Root = G.root();
    auto &Children = Root->children();

    Stack.emplace_back(Entry{ Root, true, Children.begin(), Children.end() });
    iterativeFindMaxDepth(Stack);

#else

    std::vector<Node *> RCS;
    // findMaxDepth expects the RCS to already contain the state for the first
    // element
    RCS.emplace_back(G.root());
    // run the coroutine
    findMaxDepth(RCS);

#endif
    auto End = high_resolution_clock::now();

    if (I != 0) {
      Average += (duration_cast<microseconds>(End - Start).count() / Repeat);
    }

    std::cerr << "MaxDepth: " << MaxDepth << std::endl;
    std::cerr << "Iterations: " << Iterations << std::endl;
    revng_check(MaxDepth == 31);
    revng_check(Iterations == 1227752);
  }

  std::cout << "Average: " << Average << std::endl;
  Average = 0LL;

  for (size_t I = 0; I < Repeat; I++) {
    Iterations = 0;
    MaxDepth = 0;

    size_t X = 0ULL;
    std::set<Node *> Stack;

    auto Start = high_resolution_clock::now();
#ifdef ITERATIVE

    X = iterativeFindMaxRet(G.root(), Stack);

#else

    X = findMaxDepthRet(G.root(), Stack);

#endif
    auto End = high_resolution_clock::now();

    if (I != 0) {
      Average += (duration_cast<microseconds>(End - Start).count() / Repeat);
    }

    revng_check(X == 34);
  }

  std::cout << "Average: " << Average << std::endl;

  int Result = 0;
  accumulateSums(7, Result);
  std::cout << "Result: " << Result << std::endl;
  revng_check(Result == 28);

  return 0;
}
