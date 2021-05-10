// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file    network_graph.h
 * @date    12 May 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Graph Core Class for Neural Network
 *
 */

#include <algorithm>
#include <sstream>

#include <graph_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

void GraphCore::removeEdges() {
  /**
   * remove all edges to save memory.
   * NodeList is kept for now for O(1) access of nodes by idx.
   */
  for (unsigned int i = 0; i < adj.size(); ++i) {
    /**
     * As this resize is guaranteed to not insert new elements, create a
     * default element needed by resize.
     */
    adj[i].resize(1);
  }
}

void GraphCore::addGraphNode(std::shared_ptr<GraphNode> node) {
  node->setIndex(adj.size());
  adj.push_back(std::list<std::shared_ptr<GraphNode>>({node}));
}

std::shared_ptr<GraphNode> &GraphCore::getGraphNode(unsigned int ith) {
  if (ith >= size())
    throw std::invalid_argument("Exceed total number of nodes");

  if (adj[ith].front()->getIndex() != ith)
    throw std::runtime_error("Graph internal index mismatch");

  return adj[ith].front();
}

std::shared_ptr<GraphNode> &GraphCore::getSortedGraphNode(unsigned int ith) {
  if (ith >= getSorted().size())
    throw std::invalid_argument("Exceed total number of nodes");

  return getSorted()[ith];
}

void GraphCore::topologicalSortUtil(
  unsigned int ith, std::vector<bool> &visited,
  std::stack<std::shared_ptr<GraphNode>> &Stack) {
  visited[ith] = true;

  std::list<std::shared_ptr<GraphNode>>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    auto index = (*i)->getIndex();
    if (!visited[index])
      topologicalSortUtil(index, visited, Stack);
  }

  Stack.push(getGraphNode(ith));
}

void GraphCore::topologicalSort() {
  std::stack<std::shared_ptr<GraphNode>> Stack;
  std::vector<bool> visited(adj.size());
  Sorted.clear();

  std::fill(visited.begin(), visited.end(), false);

  // Quite likely this is not needed - verify this
  // TODO : After make node list of graph, we have to find root. (That means it
  // should be the only one input for now.). Need to support multiple input and
  // support search.

  for (unsigned int i = 0; i < adj.size(); ++i) {
    if (visited[i] == false) {
      topologicalSortUtil(i, visited, Stack);
    }
  }

  while (Stack.empty() == false) {
    Sorted.push_back(Stack.top());
    Stack.pop();
  }
}

std::shared_ptr<GraphNode> &GraphCore::getGraphNode(const std::string &name) {
  for (auto &lnode_list : adj) {
    auto &lnode = lnode_list.front();
    if (lnode->getName() == name)
      return lnode;
  }

  std::stringstream ss;
  ss << "Cannot find graph node: " << name;
  throw std::invalid_argument(ss.str());
}

void GraphCore::addEdge(unsigned int ith, std::shared_ptr<GraphNode> &node) {
  if (ith >= adj.size())
    throw std::invalid_argument("Exceed total number of nodes");

  adj[ith].push_back(node);
}

std::vector<std::shared_ptr<GraphNode>> GraphCore::getGraphNodes() const {
  std::vector<std::shared_ptr<GraphNode>> ret;
  if (!Sorted.empty()) {
    std::transform(Sorted.begin(), Sorted.end(), std::back_inserter(ret),
                   [](auto const &elem) { return elem; });
  } else {
    std::transform(adj.begin(), adj.end(), std::back_inserter(ret),
                   [](auto const &elem) { return elem.front(); });
  }

  return ret;
}

void GraphCore::addNode(std::shared_ptr<GraphNode> node) {
  /** Ensure that the node has a name and is unique */
  ensureName(node);

  /** Insert the node to the graph */
  addGraphNode(node);
}

const std::vector<std::shared_ptr<GraphNode>> &GraphCore::getSorted() const {
  if (Sorted.empty())
    throw std::runtime_error("Cannot get sorted graph before topologicalSort");

  return Sorted;
}

std::vector<std::shared_ptr<GraphNode>> &GraphCore::getSorted() {
  if (Sorted.empty())
    throw std::runtime_error("Cannot get sorted graph before topologicalSort");

  return Sorted;
}

void GraphCore::ensureName(std::shared_ptr<GraphNode> &node,
                           const std::string &prefix,
                           const std::string &postfix, bool force_rename) {
  std::string orig_name = node->getName();
  bool orig_name_empty = orig_name.empty();
  /** If node already has name which is unique and valid, and force is
   * disabled, then nothing to do.
   */
  if (!orig_name_empty && !force_rename &&
      node_names.end() == node_names.find(orig_name)) {
    node_names.insert(orig_name);
    return;
  }

  /** If just prefix with node name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name + postfix;
    if (node_names.find(direct_name) == node_names.end()) {
      node->setName(direct_name);
      node_names.insert(direct_name);
      return;
    }
  }

  std::set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty) {
    orig_name = node->getType();
  }

  std::string direct_name = prefix + orig_name + postfix;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = node_names.find(name);
  } while (iter != node_names.end());

  node->setName(name);
  node_names.insert(name);
}

} /* namespace nntrainer */