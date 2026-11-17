// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title RROOTBridge
 * @notice Enterprise Multi-Sig Cross-Chain Bridge for Blackroot rROOT ↔ Swarm ROOT.
 * @dev Enforces:
 *      1. M-of-N threshold multisig via EIP-712 structured typed data.
 *      2. Sorted unique signer verification preventing duplicate signature replay.
 *      3. On-chain rolling 24-hour withdrawal volume rate limits.
 *      4. Withdrawal deadline expiration.
 *      5. EIP-2612 gasless deposit via `depositWithPermit`.
 *      6. Explicit fee telemetry and routing to a dedicated fee treasury.
 *      7. Full administrative circuit breakers (Pausable + Blacklist).
 */

interface IRROOTToken {
    function mint(address to, uint256 value) external;
    function burnFrom(address account, uint256 value) external;
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;
}

contract RROOTBridge {
    IRROOTToken public immutable token;
    address public owner;
    address public feeTreasury;
    bool public paused;

    // Bridge fee in basis points (30 = 0.30% per root_tokenomics.yaml)
    uint256 public bridgeFeeBasisPoints = 30;

    // Defense-in-depth: On-chain rolling daily rate limiting
    uint256 public dailyLimit = 25_000 * 1e6; // 25,000 rROOT (6 decimals)
    uint256 public currentDayVolume;
    uint256 public lastDayResetTimestamp;

    // Multisig validator parameters
    uint256 public validatorThreshold;
    address[] public validatorList;
    mapping(address => bool) public isValidator;

    // Nonce and blacklist registries
    uint256 public depositNonce;
    mapping(uint256 => bool) public processedWithdrawals;
    mapping(address => bool) public blacklisted;

    // EIP-712 Domain and Struct Types
    bytes32 public immutable DOMAIN_SEPARATOR;
    // keccak256("Withdrawal(address recipient,uint256 amount,uint256 nonce,uint256 deadline)")
    bytes32 public constant WITHDRAWAL_TYPEHASH = 0x8d5c404cfbbbe61b17d3d81b373cf58bb6199343729fc9cfd04260fa1f974eb4;

    // Events
    event Deposit(
        address indexed sender,
        bytes32 indexed agentId,
        uint256 grossAmount,
        uint256 feeAmount,
        uint256 netAmount,
        uint256 nonce
    );
    event Withdrawal(
        address indexed recipient,
        uint256 amount,
        uint256 nonce
    );
    event ValidatorAdded(address indexed validator);
    event ValidatorRemoved(address indexed validator);
    event ThresholdUpdated(uint256 newThreshold);
    event DailyLimitUpdated(uint256 newLimit);
    event FeeTreasuryUpdated(address indexed newTreasury);
    event FeeBasisPointsUpdated(uint256 newFeeBps);
    event BlacklistUpdated(address indexed account, bool isBlacklisted);
    event Paused(address account);
    event Unpaused(address account);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "Bridge: caller is not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Bridge: paused");
        _;
    }

    modifier notBlacklisted(address account) {
        require(!blacklisted[account], "Bridge: account blacklisted");
        _;
    }

    constructor(
        address _token,
        address _feeTreasury,
        address _initialOwner,
        address[] memory _initialValidators,
        uint256 _threshold
    ) {
        require(_token != address(0), "Bridge: zero token address");
        require(_feeTreasury != address(0), "Bridge: zero treasury address");
        require(_initialOwner != address(0), "Bridge: zero owner address");
        require(_threshold > 0 && _threshold <= _initialValidators.length, "Bridge: invalid threshold");

        token = IRROOTToken(_token);
        feeTreasury = _feeTreasury;
        owner = _initialOwner;
        validatorThreshold = _threshold;
        lastDayResetTimestamp = block.timestamp;

        for (uint256 i = 0; i < _initialValidators.length; i++) {
            address v = _initialValidators[i];
            require(v != address(0), "Bridge: zero validator address");
            require(!isValidator[v], "Bridge: duplicate validator");
            isValidator[v] = true;
            validatorList.push(v);
            emit ValidatorAdded(v);
        }

        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes("RROOTBridge")),
                keccak256(bytes("1")),
                block.chainid,
                address(this)
            )
        );
    }

    // ── User Inbound Actions ──────────────────────────────────────────────────

    function deposit(uint256 amount, bytes32 agentId)
        external
        whenNotPaused
        notBlacklisted(msg.sender)
    {
        _processDeposit(msg.sender, amount, agentId);
    }

    function depositWithPermit(
        uint256 amount,
        bytes32 agentId,
        uint256 permitDeadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external whenNotPaused notBlacklisted(msg.sender) {
        token.permit(msg.sender, address(this), amount, permitDeadline, v, r, s);
        _processDeposit(msg.sender, amount, agentId);
    }

    function _processDeposit(address sender, uint256 grossAmount, bytes32 agentId) internal {
        require(grossAmount > 0, "Bridge: amount must be > 0");

        uint256 fee = (grossAmount * bridgeFeeBasisPoints) / 10000;
        uint256 netAmount = grossAmount - fee;
        require(netAmount > 0, "Bridge: net amount is 0 after fee");

        // Explicit fee routing: route fee to feeTreasury, burn netAmount from circulation
        if (fee > 0) {
            token.transferFrom(sender, feeTreasury, fee);
        }
        token.burnFrom(sender, netAmount);

        uint256 nonce = depositNonce++;
        emit Deposit(sender, agentId, grossAmount, fee, netAmount, nonce);
    }

    // ── User Outbound Actions ─────────────────────────────────────────────────

    struct SignatureRSV {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    function withdraw(
        address recipient,
        uint256 amount,
        uint256 nonce,
        uint256 deadline,
        SignatureRSV[] calldata signatures
    ) external whenNotPaused notBlacklisted(recipient) {
        require(block.timestamp <= deadline, "Bridge: withdrawal expired");
        require(!processedWithdrawals[nonce], "Bridge: nonce already processed");
        require(recipient != address(0), "Bridge: zero recipient address");
        require(amount > 0, "Bridge: zero amount");
        require(signatures.length >= validatorThreshold, "Bridge: insufficient signatures");

        // Check on-chain rolling daily rate limit
        _enforceDailyLimit(amount);

        // Verify EIP-712 structured multi-signatures
        bytes32 structHash = keccak256(
            abi.encode(WITHDRAWAL_TYPEHASH, recipient, amount, nonce, deadline)
        );
        bytes32 digest = keccak256(
            abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash)
        );

        address lastSigner = address(0);
        uint256 validSignatures = 0;

        for (uint256 i = 0; i < signatures.length; i++) {
            address recovered = ecrecover(digest, signatures[i].v, signatures[i].r, signatures[i].s);
            require(recovered > lastSigner, "Bridge: signers not strictly ascending/unique");
            lastSigner = recovered;

            if (isValidator[recovered]) {
                validSignatures++;
                if (validSignatures >= validatorThreshold) {
                    break;
                }
            }
        }

        require(validSignatures >= validatorThreshold, "Bridge: validator threshold not met");

        // Record execution and mint backed rROOT to recipient
        processedWithdrawals[nonce] = true;
        token.mint(recipient, amount);

        emit Withdrawal(recipient, amount, nonce);
    }

    function _enforceDailyLimit(uint256 amount) internal {
        if (block.timestamp >= lastDayResetTimestamp + 1 days) {
            currentDayVolume = 0;
            lastDayResetTimestamp = block.timestamp;
        }
        require(currentDayVolume + amount <= dailyLimit, "Bridge: daily withdrawal limit exceeded");
        unchecked {
            currentDayVolume += amount;
        }
    }

    // ── Administration & Emergency Controls ───────────────────────────────────

    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    function setBlacklist(address account, bool status) external onlyOwner {
        blacklisted[account] = status;
        emit BlacklistUpdated(account, status);
    }

    function setFeeTreasury(address newTreasury) external onlyOwner {
        require(newTreasury != address(0), "Bridge: zero treasury address");
        feeTreasury = newTreasury;
        emit FeeTreasuryUpdated(newTreasury);
    }

    function setBridgeFeeBasisPoints(uint256 newFeeBps) external onlyOwner {
        require(newFeeBps <= 1000, "Bridge: fee exceeds 10%");
        bridgeFeeBasisPoints = newFeeBps;
        emit FeeBasisPointsUpdated(newFeeBps);
    }

    function setDailyLimit(uint256 newLimit) external onlyOwner {
        dailyLimit = newLimit;
        emit DailyLimitUpdated(newLimit);
    }

    function setValidatorThreshold(uint256 newThreshold) external onlyOwner {
        require(newThreshold > 0 && newThreshold <= validatorList.length, "Bridge: invalid threshold");
        validatorThreshold = newThreshold;
        emit ThresholdUpdated(newThreshold);
    }

    function addValidator(address newValidator) external onlyOwner {
        require(newValidator != address(0), "Bridge: zero validator address");
        require(!isValidator[newValidator], "Bridge: already validator");
        isValidator[newValidator] = true;
        validatorList.push(newValidator);
        emit ValidatorAdded(newValidator);
    }

    function removeValidator(address validator) external onlyOwner {
        require(isValidator[validator], "Bridge: not a validator");
        require(validatorList.length - 1 >= validatorThreshold, "Bridge: would breach threshold");
        isValidator[validator] = false;

        uint256 len = validatorList.length;
        for (uint256 i = 0; i < len; i++) {
            if (validatorList[i] == validator) {
                validatorList[i] = validatorList[len - 1];
                validatorList.pop();
                break;
            }
        }
        emit ValidatorRemoved(validator);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Bridge: zero owner address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }
}
