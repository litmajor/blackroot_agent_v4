# Blackroot Economic Layer X Wallet Documentation

## Overview

The Blackroot Economic Layer (BEL) is a modular, agent-driven economic simulation framework. It provides primitives for market operations, ledger management, staking, reputation, and adaptive ROI. The Wallet system extends BEL to support multi-chain asset management, cross-chain transactions, and agent-centric financial operations.

## Architecture

### Core Components

- **Ledger**: Tracks agent balances, stakes, reputation, debt, and system-wide metrics. Handles atomic transfers, staking, ROI, and bankruptcy logic.
- **Market**: Enables agents to place buy/sell orders, with escrow and atomic settlement. Integrates with the Ledger for real-time balance updates.
- **EconomicEngine**: Calculates adaptive ROI, manages supply growth, and updates scarcity factors based on system metrics.
- **WalletManager**: Manages agent wallets across multiple blockchains (Ethereum, Polygon, Arbitrum, Celo). Handles wallet creation, balance updates, and bridge route management.
- **CrossChainTransactionProcessor**: Processes cross-chain and on-chain transactions, including validation, fee calculation, and simulated execution.
- **EnhancedEconomicMixin**: Agent-side integration layer, providing high-level methods for wallet operations, bridging, cross-chain transfers, liquidity optimization, and opportunity analysis.

### Data Flow

1. **Agent Initialization**: Agents instantiate `EnhancedEconomicMixin`, which connects to BEL and WalletManager via Redis.
2. **Wallet Creation**: Agents create wallets on supported networks, paying a ROOT fee. Wallets are stored in Redis.
3. **Market Operations**: Agents interact with the Market to place/cancel orders, with escrow and settlement managed by the Ledger.
4. **Cross-Chain Transactions**: Agents submit transaction requests, which are validated and processed by the CrossChainTransactionProcessor.
5. **Liquidity Optimization**: Agents analyze and rebalance assets across chains for optimal distribution.
6. **System Maintenance**: Background tasks apply ROI, decay reputation, and update system health metrics.

## Agent Interaction

Agents interact with the system through the `EnhancedEconomicMixin`, which exposes async methods for all major operations:

- **Wallet Management**
  - `initialize_wallets(networks)`: Create wallets on specified networks.
  - `get_wallet_portfolio()`: Retrieve a comprehensive view of all wallet balances and status.
- **Market Participation**
  - `place_buy_order(amount, price)`, `place_sell_order(amount, price)`: Place orders in the market.
  - `cancel_order(order_id)`: Cancel an active order.
- **Cross-Chain Operations**
  - `bridge_assets(from_network, to_network, amount, token_address)`: Move assets between chains.
  - `send_cross_chain(to_agent_id, amount, from_network, to_network, token_address)`: Send assets to another agent across chains.
- **Liquidity Optimization**
  - `optimize_liquidity()`: Analyze asset distribution and suggest rebalancing moves.
  - `execute_liquidity_optimization()`: Execute suggested moves to balance liquidity.
- **Opportunity Analysis**
  - `get_cross_chain_opportunities()`: Discover arbitrage, yield farming, and cost-saving opportunities.

All operations are asynchronous and interact with Redis for state management and event publishing.

## BlackVault Integration

**BlackVault** is the secure key management system for all agent wallets and blockchain operations.

- **Key Storage**: When agents create wallets, private keys are generated, encrypted, and stored in BlackVault. Only a `vault_key_id` reference is kept in the wallet data; no raw keys are stored in Redis or code.
- **Access Control**: Only authorized agents/processes can retrieve keys from BlackVault, enforced via API tokens and role-based access. All key access attempts are logged for auditing.
- **Audit & Recovery**: BlackVault maintains detailed access logs and supports secure recovery workflows for lost credentials, with multi-party approval and notifications.
- **Integration Points**: WalletManager and EnhancedEconomicMixin use BlackVaultClient to retrieve keys for signing transactions. All blockchain operations (balance queries, transaction signing) use real RPC calls (e.g., Web3.py for Ethereum) and never expose raw keys.

## Security Considerations

- **Escrow & Settlement**: All market operations use escrow to ensure atomicity and prevent double-spending.
- **Bankruptcy Logic**: Agents exceeding debt thresholds are automatically marked as bankrupt, with assets liquidated and reputation decayed.
- **Fee Calculation**: All transfers and bridges apply fixed or percentage-based fees, ensuring system sustainability.
- **Async Operations**: All critical operations are async, supporting high concurrency and scalability.
- **Key Management**: All private keys are managed via BlackVault, with strict access control and audit logging.
- **Blockchain Integration**: All wallet balance queries and transaction submissions use real blockchain RPC endpoints (e.g., Web3.py for Ethereum), with secure signing using keys retrieved from BlackVault.

## Example Workflow

1. **Agent joins system**: Instantiates EnhancedEconomicMixin, connects to BEL and WalletManager.
2. **Creates wallets**: Pays ROOT fee, wallets are created on Ethereum, Polygon, Arbitrum.
3. **Participates in market**: Places buy/sell orders, with funds escrowed and settled atomically.
4. **Bridges assets**: Moves funds from Ethereum to Polygon, paying bridge fees.
5. **Optimizes liquidity**: Analyzes distribution, executes rebalancing moves.
6. **Stores keys in BlackVault**: All private keys are securely managed outside Redis.

## Extensibility

- **Add new networks**: Extend NetworkType and bridge_routes in WalletManager.
- **Integrate real blockchains**: Replace simulated balance updates with actual RPC calls and key management via BlackVault.
- **Advanced analytics**: Plug in DeFi protocol queries for real arbitrage and yield farming opportunities.

## References

- `bel_layer.py`: Core economic primitives and market logic.
- `wallet.py`: Multi-chain wallet management and agent integration.
- BlackVault (planned): Secure key and credential storage.

---

This documentation provides a full overview of the Blackroot Economic Layer X Wallet system, agent interaction patterns, and BlackVault's role in secure key management. For further details, see the code in `bel_layer.py` and `wallet.py`.
