//! `route_optimizer.rs` - Graph-Based Cross-Chain Swap Route Optimizer
//! 
//! **STAGE 10 | CHAPTER 4 | FILE 3**
//! 
//! This module implements a graph-based routing algorithm (similar to Bellman-Ford)
//! to find the most profitable cross-chain swap paths, dynamically factoring in:
//! - Real-time gas fees on each chain
//! - Bridge fees and slippage
//! - Liquidity constraints
//! - Execution latency
//! 
//! **Key Features:**
//! - Modified Bellman-Ford for negative cycle detection (arbitrage)
//! - Multi-hop route optimization
//! - Real-time cost updates
//! - Memory-efficient graph representation

use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::EPSILON;

/// Maximum number of hops to consider in route finding
const MAX_HOPS: usize = 5;

/// Maximum nodes in the graph (memory bounded)
const MAX_NODES: usize = 100;

/// Chain/DEX node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Edge represents a possible swap/bridge between two nodes
#[derive(Debug, Clone)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub token_in: String,
    pub token_out: String,
    /// Exchange rate (output per input unit)
    pub rate: f64,
    /// Fixed fee in token_in units
    pub fixed_fee: f64,
    /// Percentage fee (0.0 to 1.0)
    pub pct_fee: f64,
    /// Estimated slippage (0.0 to 1.0)
    pub slippage: f64,
    /// Gas cost in USD
    pub gas_cost_usd: f64,
    /// Bridge latency in milliseconds
    pub latency_ms: u64,
    /// Liquidity limit in USD
    pub liquidity_limit_usd: f64,
    /// Edge type
    pub edge_type: EdgeType,
}

/// Type of edge in the routing graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// DEX swap on same chain
    DexSwap,
    /// Cross-chain bridge
    Bridge,
    /// Wrapped token conversion (e.g., ETH -> WETH)
    Wrap,
    /// Stable swap (low slippage)
    StableSwap,
}

/// A complete route from source to destination
#[derive(Debug, Clone)]
pub struct Route {
    pub edges: Vec<Edge>,
    pub token_start: String,
    pub token_end: String,
    pub amount_in: f64,
    pub expected_amount_out: f64,
    pub total_fees_usd: f64,
    pub total_gas_usd: f64,
    pub total_slippage_pct: f64,
    pub total_latency_ms: u64,
    pub net_profit_usd: f64,
    pub confidence: f64,
}

/// Arbitrage cycle detected
#[derive(Debug, Clone)]
pub struct ArbitrageCycle {
    pub edges: Vec<Edge>,
    pub start_token: String,
    pub input_amount: f64,
    pub output_amount: f64,
    pub profit_pct: f64,
    pub profit_usd: f64,
    pub min_liquidity_usd: f64,
}

/// Routing graph for cross-chain swaps
pub struct RouteGraph {
    /// Adjacency list representation
    adjacency: HashMap<NodeId, Vec<usize>>,
    /// All edges stored in a flat vector
    edges: Vec<Edge>,
    /// Node metadata (chain ID, DEX name, etc.)
    node_metadata: HashMap<NodeId, NodeMetadata>,
    /// Token prices in USD for each node
    token_prices: HashMap<(NodeId, String), f64>,
}

/// Metadata for a routing node
#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub chain_id: u64,
    pub name: String,
    pub node_type: NodeType,
    pub supported_tokens: Vec<String>,
}

/// Type of node in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Chain,      // Represents a blockchain
    DEX,        // Represents a DEX on a chain
    Bridge,     // Represents a bridge protocol
    Wrapper,    // Represents a token wrapper
}

impl RouteGraph {
    /// Create a new empty routing graph
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            edges: Vec::new(),
            node_metadata: HashMap::new(),
            token_prices: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, id: NodeId, metadata: NodeMetadata) {
        if self.node_metadata.len() >= MAX_NODES {
            return; // Memory limit
        }
        self.node_metadata.insert(id, metadata);
        self.adjacency.entry(id).or_insert_with(Vec::new);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: Edge) {
        self.adjacency
            .entry(edge.from)
            .or_insert_with(Vec::new)
            .push(self.edges.len());
        self.edges.push(edge);
    }

    /// Update token price for a node
    pub fn update_token_price(&mut self, node: NodeId, token: String, price_usd: f64) {
        self.token_prices.insert((node, token), price_usd);
    }

    /// Find optimal route using modified Bellman-Ford
    /// 
    /// # Arguments
    /// * `start` - Starting node
    /// * `end` - Destination node
    /// * `token_in` - Input token
    /// * `token_out` - Output token
    /// * `amount_in` - Amount to swap
    /// 
    /// Returns: Optimal route if found
    pub fn find_optimal_route(
        &self,
        start: NodeId,
        end: NodeId,
        token_in: String,
        token_out: String,
        amount_in: f64,
    ) -> Option<Route> {
        if !self.adjacency.contains_key(&start) || !self.adjacency.contains_key(&end) {
            return None;
        }

        // Use BFS-limited search for efficiency
        let mut best_route: Option<Route> = None;
        let mut queue = VecDeque::new();

        // Initial state
        queue.push_back(RouteState {
            current_node: start,
            path_edges: Vec::new(),
            current_token: token_in.clone(),
            current_amount: amount_in,
            total_fees: 0.0,
            total_gas: 0.0,
            total_latency: 0,
            visited_nodes: HashSet::from([start]),
        });

        while let Some(state) = queue.pop_front() {
            // Check if we've reached the destination with correct token
            if state.current_node == end && state.current_token == token_out {
                let route = self.build_route(state, amount_in);
                
                if best_route.is_none() 
                    || route.expected_amount_out > best_route.as_ref().unwrap().expected_amount_out 
                {
                    best_route = Some(route);
                }
                continue;
            }

            // Skip if max hops exceeded
            if state.path_edges.len() >= MAX_HOPS {
                continue;
            }

            // Explore neighbors
            if let Some(edge_indices) = self.adjacency.get(&state.current_node) {
                for &edge_idx in edge_indices {
                    let edge = &self.edges[edge_idx];
                    
                    // Check token compatibility
                    if edge.token_in != state.current_token {
                        continue;
                    }

                    // Calculate output amount after this edge
                    let output = self.calculate_edge_output(edge, state.current_amount);
                    if output <= 0.0 {
                        continue;
                    }

                    // Check liquidity constraint
                    let output_value_usd = output * self.get_token_price(edge.to, edge.token_out.clone());
                    if output_value_usd > edge.liquidity_limit_usd {
                        continue;
                    }

                    // Avoid cycles (unless it's the final step to destination)
                    if state.visited_nodes.contains(&edge.to) && edge.to != end {
                        continue;
                    }

                    // Create new state
                    let mut new_visited = state.visited_nodes.clone();
                    new_visited.insert(edge.to);

                    queue.push_back(RouteState {
                        current_node: edge.to,
                        path_edges: {
                            let mut p = state.path_edges.clone();
                            p.push(edge_idx);
                            p
                        },
                        current_token: edge.token_out.clone(),
                        current_amount: output,
                        total_fees: state.total_fees + edge.fixed_fee + output * edge.pct_fee,
                        total_gas: state.total_gas + edge.gas_cost_usd,
                        total_latency: state.total_latency + edge.latency_ms,
                        visited_nodes: new_visited,
                    });
                }
            }
        }

        best_route
    }

    /// Calculate output amount for traversing an edge
    fn calculate_edge_output(&self, edge: &Edge, input_amount: f64) -> f64 {
        // Apply fees
        let after_fixed_fee = input_amount - edge.fixed_fee;
        if after_fixed_fee <= 0.0 {
            return 0.0;
        }

        let after_pct_fee = after_fixed_fee * (1.0 - edge.pct_fee);

        // Apply exchange rate
        let after_rate = after_pct_fee * edge.rate;

        // Apply slippage (simplified linear model)
        let slippage_factor = 1.0 - edge.slippage;

        after_rate * slippage_factor
    }

    /// Get token price in USD
    fn get_token_price(&self, node: NodeId, token: String) -> f64 {
        self.token_prices.get(&(node, token)).copied().unwrap_or(1.0)
    }

    /// Build a Route from a completed state
    fn build_route(&self, state: RouteState, amount_in: f64) -> Route {
        let edges: Vec<Edge> = state.path_edges.iter().map(|&idx| self.edges[idx].clone()).collect();
        
        let total_fees = state.total_fees;
        let total_gas = state.total_gas;
        let expected_out = state.current_amount;
        
        // Calculate total slippage (simplified)
        let ideal_out = amount_in * edges.iter().map(|e| e.rate).product::<f64>();
        let total_slippage = ((ideal_out - expected_out) / ideal_out * 100.0).max(0.0);

        // Calculate net profit (vs direct swap)
        let start_price = self.get_token_price(edges.first().unwrap().from, edges.first().unwrap().token_in.clone());
        let end_price = self.get_token_price(edges.last().unwrap().to, edges.last().unwrap().token_out.clone());
        let value_in_usd = amount_in * start_price;
        let value_out_usd = expected_out * end_price;
        let net_profit = value_out_usd - value_in_usd - total_fees - total_gas;

        // Confidence based on liquidity utilization and path length
        let liquidity_util = value_out_usd / edges.iter().map(|e| e.liquidity_limit_usd).fold(f64::MAX, f64::min);
        let path_penalty = 1.0 - (edges.len() as f64 * 0.1); // 10% penalty per hop
        let confidence = (1.0 - liquidity_util) * path_penalty;

        Route {
            edges,
            token_start: state.path_edges.first().map(|_| state.current_token.clone()).unwrap_or_default(),
            token_end: state.current_token,
            amount_in,
            expected_amount_out: expected_out,
            total_fees_usd: total_fees,
            total_gas_usd: total_gas,
            total_slippage_pct: total_slippage,
            total_latency_ms: state.total_latency,
            net_profit_usd: net_profit,
            confidence: confidence.max(0.0).min(1.0),
        }
    }

    /// Detect arbitrage cycles using Bellman-Ford
    /// 
    /// Returns all profitable cycles above the threshold
    pub fn detect_arbitrage_cycles(&self, min_profit_pct: f64) -> Vec<ArbitrageCycle> {
        let mut cycles = Vec::new();

        // For each starting node, try to find a profitable cycle
        for &start_node in self.adjacency.keys() {
            // Get tokens available at this node
            if let Some(metadata) = self.node_metadata.get(start_node) {
                for token in &metadata.supported_tokens {
                    if let Some(cycle) = self.find_cycle_from(start_node, token.clone(), min_profit_pct) {
                        cycles.push(cycle);
                    }
                }
            }
        }

        cycles
    }

    /// Find a profitable cycle starting from a specific node and token
    fn find_cycle_from(&self, start: NodeId, token: String, min_profit_pct: f64) -> Option<ArbitrageCycle> {
        let initial_amount = 1.0; // Normalize to 1 unit
        
        // DFS to find cycles
        let mut stack = VecDeque::new();
        stack.push_back(CycleState {
            current_node: start,
            path_edges: Vec::new(),
            current_amount: initial_amount,
            visited_nodes: HashSet::from([start]),
        });

        while let Some(state) = stack.pop_front() {
            if let Some(edge_indices) = self.adjacency.get(&state.current_node) {
                for &edge_idx in edge_indices {
                    let edge = &self.edges[edge_idx];
                    
                    if edge.token_in != state.path_edges.last()
                        .map(|&idx| &self.edges[idx].token_out)
                        .unwrap_or(&token)
                    {
                        continue;
                    }

                    let output = self.calculate_edge_output(edge, state.current_amount);
                    
                    // Check if we've returned to start
                    if edge.to == start && edge.token_out == token {
                        let profit_pct = ((output - initial_amount) / initial_amount) * 100.0;
                        
                        if profit_pct >= min_profit_pct {
                            let mut full_edges = state.path_edges.clone();
                            full_edges.push(edge_idx);
                            
                            let cycle_edges: Vec<Edge> = full_edges.iter().map(|&i| self.edges[i].clone()).collect();
                            let min_liq = cycle_edges.iter().map(|e| e.liquidity_limit_usd).fold(f64::MAX, f64::min);
                            
                            return Some(ArbitrageCycle {
                                edges: cycle_edges,
                                start_token: token.clone(),
                                input_amount: initial_amount,
                                output_amount: output,
                                profit_pct,
                                profit_usd: (output - initial_amount) * self.get_token_price(start, token.clone()),
                                min_liquidity_usd: min_liq,
                            });
                        }
                    } else if !state.visited_nodes.contains(&edge.to) && state.path_edges.len() < MAX_HOPS {
                        let mut new_visited = state.visited_nodes.clone();
                        new_visited.insert(edge.to);
                        
                        let mut new_path = state.path_edges.clone();
                        new_path.push(edge_idx);
                        
                        stack.push_back(CycleState {
                            current_node: edge.to,
                            path_edges: new_path,
                            current_amount: output,
                            visited_nodes: new_visited,
                        });
                    }
                }
            }
        }

        None
    }

    /// Get number of nodes in graph
    pub fn node_count(&self) -> usize {
        self.node_metadata.len()
    }

    /// Get number of edges in graph
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for RouteGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// State for route finding algorithm
#[derive(Clone)]
struct RouteState {
    current_node: NodeId,
    path_edges: Vec<usize>,
    current_token: String,
    current_amount: f64,
    total_fees: f64,
    total_gas: f64,
    total_latency: u64,
    visited_nodes: HashSet<NodeId>,
}

/// State for cycle detection
#[derive(Clone)]
struct CycleState {
    current_node: NodeId,
    path_edges: Vec<usize>,
    current_amount: f64,
    visited_nodes: HashSet<NodeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_route_graph_basic() {
        let mut graph = RouteGraph::new();

        // Add nodes
        graph.add_node(NodeId(1), NodeMetadata {
            chain_id: 1,
            name: "Ethereum".to_string(),
            node_type: NodeType::Chain,
            supported_tokens: vec!["ETH".to_string(), "USDC".to_string()],
        });

        graph.add_node(NodeId(2), NodeMetadata {
            chain_id: 56,
            name: "BSC".to_string(),
            node_type: NodeType::Chain,
            supported_tokens: vec!["BNB".to_string(), "USDC".to_string()],
        });

        // Add edge (bridge)
        graph.add_edge(Edge {
            from: NodeId(1),
            to: NodeId(2),
            token_in: "USDC".to_string(),
            token_out: "USDC".to_string(),
            rate: 0.999, // Small loss for bridge
            fixed_fee: 1.0,
            pct_fee: 0.001,
            slippage: 0.0001,
            gas_cost_usd: 10.0,
            latency_ms: 1000,
            liquidity_limit_usd: 1000000.0,
            edge_type: EdgeType::Bridge,
        });

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_edge_output_calculation() {
        let graph = RouteGraph::new();
        
        let edge = Edge {
            from: NodeId(1),
            to: NodeId(2),
            token_in: "A".to_string(),
            token_out: "B".to_string(),
            rate: 1.0,
            fixed_fee: 0.0,
            pct_fee: 0.003, // 0.3%
            slippage: 0.001, // 0.1%
            gas_cost_usd: 0.0,
            latency_ms: 0,
            liquidity_limit_usd: 1000000.0,
            edge_type: EdgeType::DexSwap,
        };

        let output = graph.calculate_edge_output(&edge, 1000.0);
        
        // Should be approximately 1000 * (1 - 0.003) * (1 - 0.001) = 996.003
        assert!(output > 995.0 && output < 997.0);
    }
}
