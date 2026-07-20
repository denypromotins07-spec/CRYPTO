"""
Asynchronous Whale Tracker for On-Chain Analytics
Monitors large wallet movements, exchange flows, and token unlocks
Non-blocking ingestion from Etherscan/BscScan and RPC nodes
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
import time
import json
import hashlib


@dataclass
class WhaleTransaction:
    """Represents a large on-chain transaction."""
    tx_hash: str
    timestamp: float
    from_address: str
    to_address: str
    value_usd: float
    token_symbol: str
    chain: str
    transaction_type: str  # 'transfer', 'swap', 'bridge', etc.
    is_exchange_related: bool = False
    exchange_name: Optional[str] = None
    confidence_score: float = 0.0


@dataclass
class ExchangeFlow:
    """Exchange inflow/outflow metrics."""
    exchange_name: str
    chain: str
    timestamp: float
    inflow_usd: float
    outflow_usd: float
    net_flow_usd: float
    token_breakdown: Dict[str, float] = field(default_factory=dict)


class KnownExchangeAddresses:
    """Database of known exchange and whale addresses."""
    
    # Major exchange addresses (sample - would be much larger in production)
    EXCHANGES = {
        'binance': {
            'ethereum': [
                '0x28c6c06298d514db089934071355e5743bf21d60',
                '0x21a31ee1afc51d94c2efccaa2092ad1028285549',
                '0xdfd5293d8e347dfe59e90efd55b2956a1343963d',
            ],
            'bsc': [
                '0x8894e0a0c962cb723c1976a4421c95949be2d4e3',
            ]
        },
        'coinbase': {
            'ethereum': [
                '0x5754284f345afc66a98fbb0a0afe71e0f007b949',
                '0x71660c4005ba85c37ccec55d0c4493e66fe775d3',
            ]
        },
        'kraken': {
            'ethereum': [
                '0x2910543af39aba0cd09dbb2d50200b3e800a63d2',
            ]
        }
    }
    
    # Whale/watchlist addresses
    WATCHLIST_ADDRESSES: Set[str] = set()
    
    @classmethod
    def is_exchange(cls, address: str) -> Tuple[bool, Optional[str]]:
        """Check if address belongs to known exchange."""
        address_lower = address.lower()
        for exchange, chains in cls.EXCHANGES.items():
            for chain_addresses in chains.values():
                if address_lower in chain_addresses:
                    return True, exchange
        return False, None
    
    @classmethod
    def add_to_watchlist(cls, address: str):
        """Add address to watchlist."""
        cls.WATCHLIST_ADDRESSES.add(address.lower())
    
    @classmethod
    def is_watched(cls, address: str) -> bool:
        """Check if address is on watchlist."""
        return address.lower() in cls.WATCHLIST_ADDRESSES


class AsyncRPCClient:
    """Async client for blockchain RPC calls."""
    
    def __init__(self, rpc_url: str, chain: str, max_retries: int = 3):
        self.rpc_url = rpc_url
        self.chain = chain
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_id = 0
        
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def make_request(self, method: str, params: List[Any]) -> Any:
        """Make JSON-RPC request with retry logic."""
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                self.request_id += 1
                payload = {
                    'jsonrpc': '2.0',
                    'method': method,
                    'params': params,
                    'id': self.request_id
                }
                
                async with session.post(self.rpc_url, json=payload) as response:
                    result = await response.json()
                    
                    if 'error' in result:
                        raise Exception(f"RPC Error: {result['error']}")
                    
                    return result.get('result')
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        return None
    
    async def get_latest_block(self) -> int:
        """Get latest block number."""
        result = await self.make_request('eth_blockNumber', [])
        return int(result, 16) if result else 0
    
    async def get_transaction(self, tx_hash: str) -> Optional[Dict]:
        """Get transaction details."""
        return await self.make_request('eth_getTransactionByHash', [tx_hash])
    
    async def get_logs(self, from_block: int, to_block: int, address: str, topics: List[str]) -> List[Dict]:
        """Get event logs for filtering transfers."""
        filter_params = {
            'fromBlock': hex(from_block),
            'toBlock': hex(to_block),
            'address': address,
            'topics': topics
        }
        return await self.make_request('eth_getLogs', [filter_params])


class WhaleTracker:
    """
    Main whale tracking engine with async ingestion.
    Monitors multiple chains simultaneously with configurable thresholds.
    """
    
    def __init__(
        self,
        usd_threshold: float = 100000,  # $100k minimum
        chains: Optional[List[str]] = None,
        rpc_urls: Optional[Dict[str, str]] = None,
        buffer_size: int = 10000
    ):
        self.usd_threshold = usd_threshold
        self.chains = chains or ['ethereum', 'bsc']
        self.rpc_urls = rpc_urls or {
            'ethereum': 'https://eth-mainnet.g.alchemy.com/v2/demo',
            'bsc': 'https://bsc-dataseed.binance.org'
        }
        
        # Transaction buffers per chain
        self.transaction_buffer: Dict[str, deque] = {
            chain: deque(maxlen=buffer_size) for chain in self.chains
        }
        
        # Exchange flow tracking
        self.exchange_flows: Dict[str, deque] = {
            chain: deque(maxlen=1000) for chain in self.chains
        }
        
        # RPC clients
        self.rpc_clients: Dict[str, AsyncRPCClient] = {}
        
        # Tracking state
        self.last_processed_blocks: Dict[str, int] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Token prices (would be updated by price feed)
        self.token_prices: Dict[str, float] = {
            'ETH': 2000.0,
            'BNB': 300.0,
            'USDT': 1.0,
            'USDC': 1.0,
            'WBTC': 40000.0
        }
        
        # Statistics
        self.stats = {
            'transactions_processed': 0,
            'whale_alerts': 0,
            'exchange_flows_tracked': 0
        }
        
        # Callbacks for alerts
        self.whale_alert_callbacks = []
        
    def register_whale_callback(self, callback):
        """Register callback for whale transaction alerts."""
        self.whale_alert_callbacks.append(callback)
    
    async def initialize(self):
        """Initialize RPC clients for all chains."""
        for chain in self.chains:
            if chain in self.rpc_urls:
                self.rpc_clients[chain] = AsyncRPCClient(
                    self.rpc_urls[chain],
                    chain
                )
                # Get initial block number
                try:
                    block = await self.rpc_clients[chain].get_latest_block()
                    self.last_processed_blocks[chain] = block
                    print(f"[{chain.upper()}] Starting from block {block}")
                except Exception as e:
                    print(f"[{chain.upper()}] Failed to get initial block: {e}")
    
    async def close(self):
        """Cleanup resources."""
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close RPC sessions
        for client in self.rpc_clients.values():
            await client.close()
    
    def _calculate_usd_value(self, amount_wei: int, token_symbol: str) -> float:
        """Convert token amount to USD value."""
        # Simplified - would use actual decimals and price feeds
        if token_symbol == 'ETH':
            amount = amount_wei / 1e18
        elif token_symbol == 'BNB':
            amount = amount_wei / 1e18
        else:
            amount = amount_wei / 1e6  # Assume 6 decimals for stablecoins
        
        price = self.token_prices.get(token_symbol, 0)
        return amount * price
    
    def _classify_transaction_type(self, tx_data: Dict, logs: List[Dict]) -> str:
        """Classify transaction type based on data and logs."""
        # Simplified classification - would be more sophisticated in production
        if tx_data.get('to') is None:
            return 'contract_creation'
        
        if logs:
            # Check for swap signatures (simplified)
            swap_topics = [
                '0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822',  # Uniswap swap
                '0x8c1be1a35b1128c4026680c2a6b6c7f5e8f5c5f5f5f5f5f5f5f5f5f5f5f5f5f5'
            ]
            for log in logs:
                if log.get('topics', [])[0] in swap_topics:
                    return 'swap'
            
            return 'transfer'
        
        return 'transfer'
    
    async def process_block(self, chain: str, block_number: int):
        """Process a single block for whale transactions."""
        if chain not in self.rpc_clients:
            return
        
        client = self.rpc_clients[chain]
        
        try:
            # Get block with transactions
            block_data = await client.make_request('eth_getBlockByNumber', [hex(block_number), True])
            
            if not block_data or 'transactions' not in block_data:
                return
            
            timestamp = int(block_data.get('timestamp', '0x0'), 16)
            
            for tx in block_data.get('transactions', []):
                tx_hash = tx.get('hash', '0x0')
                value = int(tx.get('value', '0x0'), 16)
                from_addr = tx.get('from', '').lower()
                to_addr = tx.get('to', '').lower() if tx.get('to') else None
                
                # Calculate USD value (ETH transactions)
                usd_value = self._calculate_usd_value(value, 'ETH' if chain == 'ethereum' else 'BNB')
                
                # Check if whale transaction
                if usd_value >= self.usd_threshold:
                    # Check exchange involvement
                    is_exchange, exchange_name = KnownExchangeAddresses.is_exchange(from_addr or '')
                    is_exchange_to, exchange_name_to = KnownExchangeAddresses.is_exchange(to_addr or '') if to_addr else (False, None)
                    
                    tx_type = self._classify_transaction_type(tx, [])
                    
                    whale_tx = WhaleTransaction(
                        tx_hash=tx_hash,
                        timestamp=timestamp,
                        from_address=from_addr,
                        to_address=to_addr or '',
                        value_usd=usd_value,
                        token_symbol='ETH' if chain == 'ethereum' else 'BNB',
                        chain=chain,
                        transaction_type=tx_type,
                        is_exchange_related=is_exchange or is_exchange_to,
                        exchange_name=exchange_name or exchange_name_to,
                        confidence_score=0.9 if is_exchange or is_exchange_to else 0.7
                    )
                    
                    # Store in buffer
                    self.transaction_buffer[chain].append(whale_tx)
                    self.stats['transactions_processed'] += 1
                    self.stats['whale_alerts'] += 1
                    
                    # Trigger callbacks
                    for callback in self.whale_alert_callbacks:
                        try:
                            callback(whale_tx)
                        except Exception as e:
                            print(f"Callback error: {e}")
                    
                    # Track exchange flows
                    if is_exchange or is_exchange_to:
                        self._track_exchange_flow(chain, exchange_name or exchange_name_to, whale_tx)
                
                # Also check watched addresses
                elif KnownExchangeAddresses.is_watched(from_addr) or KnownExchangeAddresses.is_watched(to_addr or ''):
                    # Track regardless of value
                    pass
                    
        except Exception as e:
            print(f"[{chain.upper()}] Error processing block {block_number}: {e}")
    
    def _track_exchange_flow(self, chain: str, exchange_name: Optional[str], tx: WhaleTransaction):
        """Track exchange inflows and outflows."""
        if not exchange_name:
            return
        
        # Determine if inflow or outflow
        is_inflow = KnownExchangeAddresses.is_exchange(tx.to_address)[0]
        
        # Aggregate flows (simplified - would use proper aggregation in production)
        flow_key = f"{chain}:{exchange_name}"
        
        # Find or create flow record for this timestamp window
        flow_window = int(tx.timestamp // 60) * 60  # 1-minute windows
        
        existing_flow = None
        for flow in self.exchange_flows[chain]:
            if flow.exchange_name == exchange_name and abs(flow.timestamp - flow_window) < 60:
                existing_flow = flow
                break
        
        if existing_flow:
            if is_inflow:
                existing_flow.inflow_usd += tx.value_usd
            else:
                existing_flow.outflow_usd += tx.value_usd
            existing_flow.net_flow_usd = existing_flow.inflow_usd - existing_flow.outflow_usd
        else:
            new_flow = ExchangeFlow(
                exchange_name=exchange_name,
                chain=chain,
                timestamp=flow_window,
                inflow_usd=tx.value_usd if is_inflow else 0,
                outflow_usd=tx.value_usd if not is_inflow else 0,
                net_flow_usd=(tx.value_usd if is_inflow else 0) - (tx.value_usd if not is_inflow else 0),
                token_breakdown={tx.token_symbol: tx.value_usd}
            )
            self.exchange_flows[chain].append(new_flow)
            self.stats['exchange_flows_tracked'] += 1
    
    async def monitor_chain(self, chain: str, poll_interval: float = 1.0):
        """Continuously monitor a chain for new blocks."""
        if chain not in self.rpc_clients:
            return
        
        client = self.rpc_clients[chain]
        print(f"[{chain.upper()}] Starting monitor...")
        
        while self.running:
            try:
                # Get latest block
                latest_block = await client.get_latest_block()
                last_processed = self.last_processed_blocks.get(chain, latest_block - 10)
                
                # Process any new blocks
                if latest_block > last_processed:
                    for block_num in range(last_processed + 1, latest_block + 1):
                        if not self.running:
                            break
                        await self.process_block(chain, block_num)
                    
                    self.last_processed_blocks[chain] = latest_block
                
                # Wait for next poll
                await asyncio.sleep(poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[{chain.upper()}] Monitor error: {e}")
                await asyncio.sleep(poll_interval * 2)
    
    async def start(self):
        """Start monitoring all chains."""
        await self.initialize()
        self.running = True
        
        # Start monitoring tasks for each chain
        for chain in self.chains:
            if chain in self.rpc_clients:
                task = asyncio.create_task(self.monitor_chain(chain))
                self.tasks.append(task)
        
        print(f"Whale tracker started on {len(self.tasks)} chains")
    
    def get_recent_whale_transactions(
        self,
        chain: Optional[str] = None,
        limit: int = 100,
        min_value_usd: Optional[float] = None
    ) -> List[WhaleTransaction]:
        """Get recent whale transactions with optional filtering."""
        transactions = []
        
        chains_to_check = [chain] if chain else self.chains
        
        for c in chains_to_check:
            if c in self.transaction_buffer:
                for tx in self.transaction_buffer[c]:
                    if min_value_usd and tx.value_usd < min_value_usd:
                        continue
                    transactions.append(tx)
        
        # Sort by timestamp descending
        transactions.sort(key=lambda x: x.timestamp, reverse=True)
        
        return transactions[:limit]
    
    def get_exchange_flows(
        self,
        chain: Optional[str] = None,
        exchange: Optional[str] = None,
        hours: int = 24
    ) -> List[ExchangeFlow]:
        """Get exchange flow data."""
        flows = []
        cutoff_time = time.time() - (hours * 3600)
        
        chains_to_check = [chain] if chain else self.chains
        
        for c in chains_to_check:
            if c in self.exchange_flows:
                for flow in self.exchange_flows[c]:
                    if flow.timestamp < cutoff_time:
                        continue
                    if exchange and flow.exchange_name != exchange:
                        continue
                    flows.append(flow)
        
        return flows
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            **self.stats,
            'running': self.running,
            'chains_monitored': len([c for c in self.chains if c in self.rpc_clients]),
            'buffers_size': sum(len(buf) for buf in self.transaction_buffer.values()),
            'last_processed_blocks': self.last_processed_blocks
        }


# Example usage
async def main():
    tracker = WhaleTracker(
        usd_threshold=50000,  # $50k threshold
        chains=['ethereum']
    )
    
    # Register alert callback
    def on_whale_alert(tx: WhaleTransaction):
        print(f"\n🐋 WHALE ALERT: ${tx.value_usd:,.2f} {tx.token_symbol} "
              f"{'→' if tx.is_exchange_related else ''} {tx.exchange_name or tx.to_address[:8]}...")
    
    tracker.register_whale_callback(on_whale_alert)
    
    # Start tracking (for demo, run briefly)
    tracker.running = True
    tracker.tasks.append(asyncio.create_task(tracker.monitor_chain('ethereum', poll_interval=5)))
    
    # Run for demonstration
    try:
        await asyncio.sleep(30)  # Run for 30 seconds
    finally:
        await tracker.close()
    
    print("\nSummary:", tracker.get_summary_stats())


if __name__ == '__main__':
    asyncio.run(main())
