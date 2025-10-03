const { ethers } = require('ethers');

// Polygon Mumbai testnet configuration
const MUMBAI_RPC_URL = process.env.MUMBAI_RPC_URL || 'https://rpc-mumbai.maticvigil.com';
const DEPLOYER_PRIVATE_KEY = process.env.DEPLOYER_PRIVATE_KEY;
const CONTRACT_ADDRESS = process.env.CONTRACT_ADDRESS;

// Smart Contract ABI (simplified for report storage)
const CONTRACT_ABI = [
  {
    "inputs": [
      {"internalType": "string", "name": "cid", "type": "string"},
      {"internalType": "string", "name": "reportHash", "type": "string"}
    ],
    "name": "storeReport",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "address", "name": "student", "type": "address"}
    ],
    "name": "getStudentReports",
    "outputs": [
      {"internalType": "string[]", "name": "", "type": "string[]"}
    ],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {"internalType": "string", "name": "cid", "type": "string"},
      {"internalType": "string", "name": "reportHash", "type": "string"}
    ],
    "name": "verifyReport",
    "outputs": [
      {"internalType": "bool", "name": "", "type": "bool"}
    ],
    "stateMutability": "view",
    "type": "function"
  }
];

/**
 * Initialize blockchain connection
 * @returns {object} - Provider and contract instances
 */
function initializeBlockchain() {
  try {
    if (!DEPLOYER_PRIVATE_KEY || !CONTRACT_ADDRESS) {
      throw new Error('Blockchain configuration missing');
    }

    const provider = new ethers.JsonRpcProvider(MUMBAI_RPC_URL);
    const wallet = new ethers.Wallet(DEPLOYER_PRIVATE_KEY, provider);
    const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, wallet);

    return { provider, wallet, contract };
  } catch (error) {
    console.error('Blockchain initialization error:', error);
    throw error;
  }
}

/**
 * Store report CID on Polygon Mumbai blockchain
 * @param {string} studentAddress - Student's wallet address
 * @param {string} cid - IPFS CID of encrypted report
 * @param {string} reportHash - Hash of report for verification
 * @returns {Promise<string>} - Transaction hash
 */
async function storeReportOnChain(studentAddress, cid, reportHash) {
  try {
    const { contract } = initializeBlockchain();

    console.log(`📝 Storing report on blockchain for ${studentAddress}`);
    
    // Estimate gas
    const gasEstimate = await contract.storeReport.estimateGas(cid, reportHash);
    
    // Send transaction
    const tx = await contract.storeReport(cid, reportHash, {
      gasLimit: gasEstimate * 120n / 100n, // Add 20% buffer
      gasPrice: ethers.parseUnits('30', 'gwei') // 30 gwei
    });

    console.log(`⏳ Transaction sent: ${tx.hash}`);
    
    // Wait for confirmation
    const receipt = await tx.wait(2); // Wait for 2 confirmations
    
    console.log(`✅ Report stored on blockchain: ${receipt.transactionHash}`);
    return receipt.transactionHash;

  } catch (error) {
    console.error('Blockchain storage error:', error);
    throw new Error(`Blockchain storage failed: ${error.message}`);
  }
}

/**
 * Get all report CIDs for a student
 * @param {string} studentAddress - Student's wallet address
 * @returns {Promise<string[]>} - Array of CIDs
 */
async function getStudentReports(studentAddress) {
  try {
    const { contract } = initializeBlockchain();
    
    const reports = await contract.getStudentReports(studentAddress);
    return reports;
  } catch (error) {
    console.error('Error fetching student reports:', error);
    throw new Error(`Failed to fetch reports: ${error.message}`);
  }
}

/**
 * Verify report integrity on blockchain
 * @param {string} cid - IPFS CID
 * @param {string} reportHash - Hash to verify
 * @returns {Promise<boolean>} - True if verified
 */
async function verifyReportOnChain(cid, reportHash) {
  try {
    const { contract } = initializeBlockchain();
    
    const isValid = await contract.verifyReport(cid, reportHash);
    return isValid;
  } catch (error) {
    console.error('Verification error:', error);
    return false;
  }
}

/**
 * Get current gas price for Mumbai network
 * @returns {Promise<bigint>} - Gas price in wei
 */
async function getCurrentGasPrice() {
  try {
    const { provider } = initializeBlockchain();
    const gasPrice = await provider.getFeeData();
    return gasPrice.gasPrice || ethers.parseUnits('30', 'gwei');
  } catch (error) {
    console.error('Gas price fetch error:', error);
    return ethers.parseUnits('30', 'gwei'); // Fallback
  }
}

module.exports = {
  storeReportOnChain,
  getStudentReports,
  verifyReportOnChain,
  getCurrentGasPrice
};