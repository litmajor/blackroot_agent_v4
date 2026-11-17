// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title RROOTToken
 * @notice Blackroot Wrapped ROOT Token (rROOT).
 * @dev 1:1 backed on-chain synthetic representation of internal swarm ROOT credits.
 *      Decimals set to 6 to maintain 1:1 parity with the internal BEL v3 micro-settlement engine.
 *      Includes ERC-20, ERC20Burnable, and EIP-2612 Permit for gasless bridge deposits.
 */

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

interface IERC20Permit {
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;
    function nonces(address owner) external view returns (uint256);
    function DOMAIN_SEPARATOR() external view returns (bytes32);
}

contract RROOTToken is IERC20, IERC20Permit {
    string public constant name = "Blackroot Wrapped ROOT";
    string public constant symbol = "rROOT";
    uint8 public constant decimals = 6;

    uint256 private _totalSupply;
    address public owner;
    address public bridgeContract;

    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    mapping(address => uint256) public override nonces;

    bytes32 public immutable override DOMAIN_SEPARATOR;
    // keccak256("Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)")
    bytes32 public constant PERMIT_TYPEHASH = 0x6e71edae12b1b97f4d1f60370fef10105fa2faae0126114a169c64845d6126c9;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event BridgeContractUpdated(address indexed previousBridge, address indexed newBridge);

    modifier onlyOwner() {
        require(msg.sender == owner, "RROOT: caller is not owner");
        _;
    }

    modifier onlyBridge() {
        require(msg.sender == bridgeContract, "RROOT: caller is not bridge");
        _;
    }

    constructor(address initialOwner) {
        require(initialOwner != address(0), "RROOT: zero owner address");
        owner = initialOwner;
        emit OwnershipTransferred(address(0), initialOwner);

        DOMAIN_SEPARATOR = keccak256(
            abi.encode(
                keccak256("EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"),
                keccak256(bytes(name)),
                keccak256(bytes("1")),
                block.chainid,
                address(this)
            )
        );
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "RROOT: new owner is zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setBridge(address newBridge) external onlyOwner {
        require(newBridge != address(0), "RROOT: bridge is zero address");
        emit BridgeContractUpdated(bridgeContract, newBridge);
        bridgeContract = newBridge;
    }

    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }

    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }

    function transfer(address to, uint256 value) external override returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }

    function allowance(address accountOwner, address spender) external view override returns (uint256) {
        return _allowances[accountOwner][spender];
    }

    function approve(address spender, uint256 value) external override returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external override returns (bool) {
        uint256 currentAllowance = _allowances[from][msg.sender];
        if (currentAllowance != type(uint256).max) {
            require(currentAllowance >= value, "RROOT: insufficient allowance");
            unchecked {
                _approve(from, msg.sender, currentAllowance - value);
            }
        }
        _transfer(from, to, value);
        return true;
    }

    function mint(address to, uint256 value) external onlyBridge {
        require(to != address(0), "RROOT: mint to zero address");
        _totalSupply += value;
        unchecked {
            _balances[to] += value;
        }
        emit Transfer(address(0), to, value);
    }

    function burn(uint256 value) external {
        _burn(msg.sender, value);
    }

    function burnFrom(address account, uint256 value) external {
        uint256 currentAllowance = _allowances[account][msg.sender];
        if (msg.sender != bridgeContract && currentAllowance != type(uint256).max) {
            require(currentAllowance >= value, "RROOT: burn exceeds allowance");
            unchecked {
                _approve(account, msg.sender, currentAllowance - value);
            }
        }
        _burn(account, value);
    }

    function permit(
        address accountOwner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external override {
        require(block.timestamp <= deadline, "RROOT: permit expired");

        bytes32 structHash = keccak256(
            abi.encode(
                PERMIT_TYPEHASH,
                accountOwner,
                spender,
                value,
                nonces[accountOwner]++,
                deadline
            )
        );

        bytes32 hash = keccak256(
            abi.encodePacked("\x19\x01", DOMAIN_SEPARATOR, structHash)
        );

        address signer = ecrecover(hash, v, r, s);
        require(signer != address(0) && signer == accountOwner, "RROOT: invalid permit signature");

        _approve(accountOwner, spender, value);
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(from != address(0), "RROOT: transfer from zero");
        require(to != address(0), "RROOT: transfer to zero");
        uint256 fromBalance = _balances[from];
        require(fromBalance >= value, "RROOT: balance exceeded");
        unchecked {
            _balances[from] = fromBalance - value;
            _balances[to] += value;
        }
        emit Transfer(from, to, value);
    }

    function _burn(address account, uint256 value) internal {
        require(account != address(0), "RROOT: burn from zero");
        uint256 accountBalance = _balances[account];
        require(accountBalance >= value, "RROOT: burn exceeds balance");
        unchecked {
            _balances[account] = accountBalance - value;
            _totalSupply -= value;
        }
        emit Transfer(account, address(0), value);
    }

    function _approve(address accountOwner, address spender, uint256 value) internal {
        require(accountOwner != address(0), "RROOT: approve from zero");
        require(spender != address(0), "RROOT: approve to zero");
        _allowances[accountOwner][spender] = value;
        emit Approval(accountOwner, spender, value);
    }
}
