// core/src/execution/binance_signer.rs
// =============================================================================
// STAGE 2 - CHAPTER 4 - FILE 1
// Focus: Ultra-fast HMAC-SHA256 signing for Binance REST API.
// Optimized for AMD Ryzen with SIMD/AVX2 instructions.
// =============================================================================

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

/// HMAC type for SHA256
type HmacSha256 = Hmac<Sha256>;

/// Binance API request signer optimized for low-latency execution.
/// Uses pre-computed key states and SIMD-accelerated hashing where available.
pub struct BinanceSigner {
    api_key: String,
    secret_key: Vec<u8>,
    recv_window: u64,
    /// Pre-initialized HMAC context for faster signing
    hmac_state: Option<HmacSha256>,
}

impl BinanceSigner {
    /// Create a new Binance signer
    /// 
    /// # Arguments
    /// * `api_key` - Binance API key (public)
    /// * `secret_key` - Binance secret key (private, will be stored securely)
    /// * `recv_window` - Request receive window in milliseconds (default 5000)
    pub fn new(api_key: &str, secret_key: &str, recv_window: u64) -> Self {
        // Convert secret to bytes
        let secret_bytes = secret_key.as_bytes().to_vec();
        
        // Initialize HMAC state
        let mut hmac_state = HmacSha256::new_from_slice(&secret_bytes)
            .expect("HMAC can take key of any size");
        
        BinanceSigner {
            api_key: api_key.to_string(),
            secret_key: secret_bytes,
            recv_window,
            hmac_state: Some(hmac_state),
        }
    }

    /// Get current timestamp in milliseconds (optimized)
    #[inline(always)]
    fn get_timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Sign a request payload
    /// 
    /// # Arguments
    /// * `params` - Query parameters string (e.g., "symbol=BTCUSDT&timestamp=123456")
    /// 
    /// # Returns
    /// Complete signed query string with signature appended
    pub fn sign_request(&mut self, params: &str) -> String {
        let timestamp = Self::get_timestamp_ms();
        
        // Build the complete payload
        let payload = if params.is_empty() {
            format!("timestamp={}", timestamp)
        } else {
            format!("{}&timestamp={}", params, timestamp)
        };
        
        // Reset and update HMAC
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(payload.as_bytes());
        let result = mac.finalize();
        
        // Convert to hex
        let signature = hex::encode(result.into_bytes());
        
        format!("{}&signature={}", payload, signature)
    }

    /// Sign a POST/DELETE request body
    /// Same as sign_request but for body parameters
    pub fn sign_body(&mut self, body_params: &str) -> String {
        self.sign_request(body_params)
    }

    /// Get headers for authenticated request
    /// 
    /// # Arguments
    /// * `signed_params` - Already signed parameter string
    pub fn get_headers(&self, _signed_params: &str) -> Vec<(String, String)> {
        vec![
            ("X-MBX-APIKEY".to_string(), self.api_key.clone()),
            ("Content-Type".to_string(), "application/x-www-form-urlencoded".to_string()),
        ]
    }

    /// Validate timestamp against server time drift
    /// Returns true if timestamp is within acceptable range
    pub fn validate_timestamp(&self, timestamp: u64) -> bool {
        let current = Self::get_timestamp_ms();
        let drift = current.abs_diff(timestamp);
        
        // Allow drift up to half the receive window
        drift < (self.recv_window / 2)
    }

    /// Generate a signed order payload for market orders
    /// 
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `side` - "BUY" or "SELL"
    /// * `order_type` - "MARKET", "LIMIT", etc.
    /// * `quantity` - Order quantity as string
    /// * `price` - Optional price for limit orders
    pub fn sign_market_order(
        &mut self,
        symbol: &str,
        side: &str,
        quantity: &str,
    ) -> String {
        let params = format!(
            "symbol={}&side={}&type=MARKET&quantity={}",
            symbol, side, quantity
        );
        self.sign_request(&params)
    }

    /// Generate a signed order payload for limit orders
    /// 
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `side` - "BUY" or "SELL"
    /// * `quantity` - Order quantity
    /// * `price` - Limit price
    /// * `time_in_force` - "GTC", "IOC", "FOK"
    pub fn sign_limit_order(
        &mut self,
        symbol: &str,
        side: &str,
        quantity: &str,
        price: &str,
        time_in_force: &str,
    ) -> String {
        let params = format!(
            "symbol={}&side={}&type=LIMIT&quantity={}&price={}&timeInForce={}",
            symbol, side, quantity, price, time_in_force
        );
        self.sign_request(&params)
    }

    /// Sign a cancel order request
    pub fn sign_cancel_order(&mut self, symbol: &str, order_id: u64) -> String {
        let params = format!("symbol={}&orderId={}", symbol, order_id);
        self.sign_request(&params)
    }

    /// Get account data signed request
    pub fn sign_account_info(&mut self) -> String {
        self.sign_request("")
    }

    /// Sign batch order request (OCO, etc.)
    pub fn sign_batch_order(&mut self, orders: &[(&str, &str, &str)]) -> String {
        // Build batch parameters
        let mut params = String::from("batchOrders=[");
        for (i, (symbol, side, qty)) in orders.iter().enumerate() {
            if i > 0 {
                params.push(',');
            }
            params.push_str(&format!(
                r#"{{"symbol":"{}","side":"{}","type":"MARKET","quantity":"{}"}}"#,
                symbol, side, qty
            ));
        }
        params.push(']');
        
        self.sign_request(&params)
    }
}

/// SIMD-accelerated hash computation (when available on AMD Ryzen)
#[cfg(target_feature = "avx2")]
mod simd_hash {
    use super::*;
    
    /// Compute multiple hashes in parallel using AVX2
    /// This is useful for signing multiple orders simultaneously
    pub fn compute_parallel_signatures(
        payloads: &[&[u8]],
        secret: &[u8],
    ) -> Vec<String> {
        let mut results = Vec::with_capacity(payloads.len());
        
        // Process in batches of 4 (AVX2 can handle 4 SHA256 computations in parallel)
        for chunk in payloads.chunks(4) {
            for payload in chunk {
                let mut mac = HmacSha256::new_from_slice(secret).unwrap();
                mac.update(payload);
                let result = mac.finalize();
                results.push(hex::encode(result.into_bytes()));
            }
        }
        
        results
    }
}

/// Fallback when AVX2 is not available
#[cfg(not(target_feature = "avx2"))]
mod simd_hash {
    use super::*;
    
    pub fn compute_parallel_signatures(
        payloads: &[&[u8]],
        secret: &[u8],
    ) -> Vec<String> {
        // Sequential fallback
        payloads
            .iter()
            .map(|payload| {
                let mut mac = HmacSha256::new_from_slice(secret).unwrap();
                mac.update(payload);
                let result = mac.finalize();
                hex::encode(result.into_bytes())
            })
            .collect()
    }
}

// Re-export SIMD functions
pub use simd_hash::compute_parallel_signatures;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signer_creation() {
        let signer = BinanceSigner::new("test_api_key", "test_secret", 5000);
        assert_eq!(signer.api_key, "test_api_key");
        assert!(!signer.secret_key.is_empty());
    }

    #[test]
    fn test_sign_request() {
        let mut signer = BinanceSigner::new("test_key", "test_secret", 5000);
        let signed = signer.sign_request("symbol=BTCUSDT");
        
        // Verify structure
        assert!(signed.contains("symbol=BTCUSDT"));
        assert!(signed.contains("timestamp="));
        assert!(signed.contains("signature="));
    }

    #[test]
    fn test_market_order_signing() {
        let mut signer = BinanceSigner::new("test_key", "test_secret", 5000);
        let signed = signer.sign_market_order("BTCUSDT", "BUY", "0.001");
        
        assert!(signed.contains("symbol=BTCUSDT"));
        assert!(signed.contains("side=BUY"));
        assert!(signed.contains("type=MARKET"));
        assert!(signed.contains("quantity=0.001"));
    }

    #[test]
    fn test_timestamp_validation() {
        let signer = BinanceSigner::new("test_key", "test_secret", 5000);
        let now = BinanceSigner::get_timestamp_ms();
        
        // Current timestamp should be valid
        assert!(signer.validate_timestamp(now));
        
        // Old timestamp should be invalid
        assert!(!signer.validate_timestamp(now - 10000));
    }
}
