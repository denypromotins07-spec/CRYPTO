"""
custom_execution_client.py
--------------------------
Custom Nautilus ExecutionClient that serializes commands via FlatBuffers
and pushes them to the Rust execution engine via a lock-free ring buffer.
Designed for microsecond-level order submission latency.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from nautilus_trader.core.uuid import UUID4
from nautilus_trader.execution.client import ExecutionClient
from nautilus_trader.execution.messages import (
    SubmitOrder,
    CancelOrder,
    ModifyOrder,
    CancelAllOrders,
)
from nautilus_trader.model.enums import OrderSide, OrderType, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Venue, AccountId
from nautilus_trader.model.objects import Quantity, Price

from .shared_memory import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)


# FlatBuffers-style message layout constants
MSG_SUBMIT_ORDER = 1
MSG_CANCEL_ORDER = 2
MSG_MODIFY_ORDER = 3
MSG_CANCEL_ALL = 4

COMMAND_SLOT_SIZE = 256  # Bytes per command slot


class CustomExecutionClient(ExecutionClient):
    """
    High-performance ExecutionClient for Rust backend integration.
    
    Features:
    - Lock-free command submission via ring buffer
    - FlatBuffers-style serialization for zero-copy parsing in Rust
    - Async acknowledgment handling
    - Strict memory bounds (pre-allocated buffers)
    """

    def __init__(self, venue: Venue, config: dict, loop: asyncio.AbstractEventLoop):
        super().__init__(venue=venue, config=config, loop=loop)
        
        self._venue = venue
        self._account_id: Optional[AccountId] = None
        
        # Pre-allocated command buffer (lock-free ring buffer)
        self._cmd_buffer_size = 10_000  # Max pending commands
        self._cmd_buffer: Optional[SharedMemoryRingBuffer] = None
        
        # Pending orders tracking (for reconciliation)
        self._pending_orders: Dict[str, dict] = {}
        
        self._is_connected = False
        self._ack_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """Connect to the Rust execution engine."""
        logger.info(f"Connecting CustomExecutionClient for venue {self._venue}")
        
        # Create command ring buffer (smaller than data buffer, ~25MB)
        self._cmd_buffer = SharedMemoryRingBuffer(
            name=f"nautilus_cmds_{self._venue}", 
            size_mb=25
        )
        
        self._is_connected = True
        self._ack_task = asyncio.create_task(self._acknowledgment_loop())
        
        # Generate account ID
        self._account_id = AccountId(f"{self._venue.value}-EXEC-001")
        self._generate_account_snapshot()
        
        logger.info(f"Connected to Rust execution engine for {self._venue}")

    async def disconnect(self) -> None:
        """Disconnect and cleanup."""
        self._is_connected = False
        if self._ack_task:
            self._ack_task.cancel()
            try:
                await self._ack_task
            except asyncio.CancelledError:
                pass
        
        if self._cmd_buffer:
            self._cmd_buffer.close()
            
        logger.info(f"Disconnected CustomExecutionClient for {self._venue}")

    def _generate_account_snapshot(self):
        """Generate initial account snapshot."""
        # In production, fetch real balances from exchange
        self._handle_account_state(
            account_id=self._account_id,
            venue=self._venue,
            balances=[],  # Populated from Rust
            margins=[],
            instruments=[]
        )

    async def _acknowledgment_loop(self):
        """
        Listen for order acknowledgments from Rust engine.
        Uses a separate response ring buffer.
        """
        # In production, this would read from a response buffer
        while self._is_connected:
            await asyncio.sleep(0.0001)  # Ultra-low latency poll
            # Process acks when available

    def submit_order(self, command: SubmitOrder) -> None:
        """
        Submit an order to the Rust execution engine.
        Serializes the order into a flat binary format.
        """
        if not self._is_connected:
            logger.error("Execution client not connected")
            return
        
        order = command.order
        instrument_id = order.instrument_id
        
        # Serialize order to flat buffer format
        # Layout: [msg_type(1), order_id(16), instrument_id(32), side(1), type(1), 
        #          quantity(8), price(8), stop_price(8), tif(1), post_only(1), reduce_only(1)]
        
        cmd_data = self._serialize_submit_order(order)
        
        # Write to ring buffer (lock-free)
        if self._cmd_buffer:
            self._cmd_buffer.write_command(cmd_data)
            
            # Track pending order locally
            self._pending_orders[order.client_order_id.value] = {
                'order': order,
                'ts_submitted': datetime.utcnow(),
                'status': 'PENDING'
            }
            
            logger.debug(f"Submitted order {order.client_order_id} to Rust engine")

    def cancel_order(self, command: CancelOrder) -> None:
        """Cancel an existing order."""
        if not self._is_connected:
            return
        
        cmd_data = self._serialize_cancel_order(command)
        
        if self._cmd_buffer:
            self._cmd_buffer.write_command(cmd_data)
            logger.debug(f"Cancel request for {command.order.client_order_id}")

    def modify_order(self, command: ModifyOrder) -> None:
        """Modify an existing order (price/quantity)."""
        if not self._is_connected:
            return
        
        cmd_data = self._serialize_modify_order(command)
        
        if self._cmd_buffer:
            self._cmd_buffer.write_command(cmd_data)
            logger.debug(f"Modify request for {command.order.client_order_id}")

    def cancel_all_orders(self, command: CancelAllOrders) -> None:
        """Cancel all orders for a given instrument."""
        if not self._is_connected:
            return
        
        cmd_data = self._serialize_cancel_all(command)
        
        if self._cmd_buffer:
            self._cmd_buffer.write_command(cmd_data)
            logger.info(f"Cancel all orders for {command.instrument_id}")

    def _serialize_submit_order(self, order) -> bytes:
        """Serialize SubmitOrder to flat binary format."""
        # Fixed-size buffer for zero-copy efficiency
        buf = bytearray(COMMAND_SLOT_SIZE)
        
        # Message type
        buf[0] = MSG_SUBMIT_ORDER
        
        # Order ID (UUID - 16 bytes)
        order_id_bytes = order.client_order_id.value.encode('utf-8')[:16]
        buf[1:17] = order_id_bytes.ljust(16, b'\x00')
        
        # Instrument ID (simplified - 32 bytes)
        inst_id_bytes = str(order.instrument_id).encode('utf-8')[:32]
        buf[17:49] = inst_id_bytes.ljust(32, b'\x00')
        
        # Side (1 byte): 1=Buy, 2=Sell
        buf[49] = 1 if order.side == OrderSide.BUY else 2
        
        # Order Type (1 byte): 1=Market, 2=Limit, 3=StopLimit, etc.
        type_map = {OrderType.MARKET: 1, OrderType.LIMIT: 2, OrderType.STOP_LIMIT: 3}
        buf[50] = type_map.get(order.order_type, 1)
        
        # Quantity (8 bytes float64)
        qty_bytes = struct.pack('d', float(order.quantity))
        buf[51:59] = qty_bytes
        
        # Price (8 bytes float64)
        price = float(order.price) if order.price else 0.0
        price_bytes = struct.pack('d', price)
        buf[59:67] = price_bytes
        
        # Stop Price (8 bytes float64)
        stop = float(order.trigger_price) if order.trigger_price else 0.0
        stop_bytes = struct.pack('d', stop)
        buf[67:75] = stop_bytes
        
        # TimeInForce (1 byte)
        tif_map = {TimeInForce.GTC: 1, TimeInForce.IOC: 2, TimeInForce.FOK: 3}
        buf[75] = tif_map.get(order.time_in_force, 1)
        
        # Flags (1 byte): bit0=post_only, bit1=reduce_only
        flags = 0
        if order.is_post_only:
            flags |= 0b001
        if order.is_reduce_only:
            flags |= 0b010
        buf[76] = flags
        
        return bytes(buf)

    def _serialize_cancel_order(self, command: CancelOrder) -> bytes:
        """Serialize CancelOrder to flat binary format."""
        buf = bytearray(COMMAND_SLOT_SIZE)
        buf[0] = MSG_CANCEL_ORDER
        
        order_id_bytes = command.order.client_order_id.value.encode('utf-8')[:16]
        buf[1:17] = order_id_bytes.ljust(16, b'\x00')
        
        return bytes(buf)

    def _serialize_modify_order(self, command: ModifyOrder) -> bytes:
        """Serialize ModifyOrder to flat binary format."""
        buf = bytearray(COMMAND_SLOT_SIZE)
        buf[0] = MSG_MODIFY_ORDER
        
        order_id_bytes = command.order.client_order_id.value.encode('utf-8')[:16]
        buf[1:17] = order_id_bytes.ljust(16, b'\x00')
        
        # New quantity
        if command.quantity:
            qty_bytes = struct.pack('d', float(command.quantity))
            buf[17:25] = qty_bytes
        
        # New price
        if command.price:
            price_bytes = struct.pack('d', float(command.price))
            buf[25:33] = price_bytes
        
        return bytes(buf)

    def _serialize_cancel_all(self, command: CancelAllOrders) -> bytes:
        """Serialize CancelAllOrders to flat binary format."""
        buf = bytearray(COMMAND_SLOT_SIZE)
        buf[0] = MSG_CANCEL_ALL
        
        inst_id_bytes = str(command.instrument_id).encode('utf-8')[:32]
        buf[1:33] = inst_id_bytes.ljust(32, b'\x00')
        
        return bytes(buf)


# Import struct for packing
import struct
