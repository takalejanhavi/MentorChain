const express = require('express');
const cors = require('cors');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json());

const CHAIN_FILE = path.join(__dirname, 'blockchain.json');

// ---------------------------
// Simple blockchain class
// ---------------------------
class SimpleBlockchain {
  constructor() {
    if (fs.existsSync(CHAIN_FILE)) {
      const data = JSON.parse(fs.readFileSync(CHAIN_FILE));
      this.chain = data.chain || [this.createGenesisBlock()];
      this.pendingTransactions = data.pendingTransactions || [];
    } else {
      this.chain = [this.createGenesisBlock()];
      this.pendingTransactions = [];
      this.saveChain();
    }
  }

  createGenesisBlock() {
    const timestamp = Date.now();
    const genesisData = {
      index: 0,
      timestamp,
      data: 'Genesis Block',
      previousHash: '0',
      nonce: 0
    };
    genesisData.hash = this.calculateHash(genesisData.index, genesisData.timestamp, genesisData.data, genesisData.previousHash, genesisData.nonce);
    return genesisData;
  }

  calculateHash(index, timestamp, data, previousHash, nonce = 0) {
    return crypto
      .createHash('sha256')
      .update(index + timestamp + JSON.stringify(data) + previousHash + nonce)
      .digest('hex');
  }

  getLatestBlock() {
    return this.chain[this.chain.length - 1];
  }

  addBlock(data) {
    const previousBlock = this.getLatestBlock();
    const newBlock = {
      index: previousBlock.index + 1,
      timestamp: Date.now(),
      data,
      previousHash: previousBlock.hash,
      nonce: 0
    };

    let hash = this.calculateHash(newBlock.index, newBlock.timestamp, newBlock.data, newBlock.previousHash, newBlock.nonce);
    while (!hash.startsWith('0000')) {
      newBlock.nonce++;
      hash = this.calculateHash(newBlock.index, newBlock.timestamp, newBlock.data, newBlock.previousHash, newBlock.nonce);
    }
    newBlock.hash = hash;

    this.chain.push(newBlock);
    this.saveChain();
    return newBlock;
  }

  getBlockByHash(hash) {
    return this.chain.find(block => block.hash === hash);
  }

  validateChain() {
    for (let i = 1; i < this.chain.length; i++) {
      const current = this.chain[i];
      const prev = this.chain[i - 1];
      const currentHash = this.calculateHash(current.index, current.timestamp, current.data, current.previousHash, current.nonce);
      if (current.hash !== currentHash || current.previousHash !== prev.hash) return false;
    }
    return true;
  }

  saveChain() {
    fs.writeFileSync(CHAIN_FILE, JSON.stringify({ chain: this.chain, pendingTransactions: this.pendingTransactions }, null, 2));
  }
}

// ---------------------------
// Initialize blockchain
// ---------------------------
const blockchain = new SimpleBlockchain();

// ---------------------------
// Routes
// ---------------------------

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'OK',
    message: 'Blockchain service is running',
    blocks: blockchain.chain.length,
    valid: blockchain.validateChain()
  });
});

// Store report on blockchain
app.post('/api/store', (req, res) => {
  try {
    const { reportId, userId, reportData } = req.body;
    if (!reportId || !userId || !reportData) return res.status(400).json({ success: false, error: 'Missing required fields' });

    const blockData = {
      type: 'CAREER_REPORT',
      reportId,
      userId,
      reportHash: crypto.createHash('sha256').update(JSON.stringify(reportData)).digest('hex'),
      timestamp: new Date().toISOString(),
      permissions: []
    };

    const newBlock = blockchain.addBlock(blockData);
    console.log(`📦 New report stored on blockchain: Block #${newBlock.index}`);

    res.json({
      success: true,
      transactionHash: newBlock.hash,
      blockIndex: newBlock.index,
      timestamp: newBlock.timestamp
    });
  } catch (error) {
    console.error('Blockchain storage error:', error);
    res.status(500).json({ success: false, error: 'Failed to store on blockchain' });
  }
});

// Grant permission
app.post('/api/grant-permission', (req, res) => {
  try {
    const { reportHash, studentId, psychologistId, expiresAt } = req.body;
    if (!reportHash || !studentId || !psychologistId) return res.status(400).json({ success: false, error: 'Missing required fields' });

    const blockData = {
      type: 'PERMISSION_GRANT',
      reportHash,
      studentId,
      psychologistId,
      expiresAt,
      timestamp: new Date().toISOString()
    };

    const newBlock = blockchain.addBlock(blockData);
    console.log(`🔐 Permission granted on blockchain: Block #${newBlock.index}`);

    res.json({ success: true, transactionHash: newBlock.hash, blockIndex: newBlock.index });
  } catch (error) {
    console.error('Permission grant error:', error);
    res.status(500).json({ success: false, error: 'Failed to grant permission' });
  }
});

// Revoke permission
app.post('/api/revoke-permission', (req, res) => {
  try {
    const { reportHash, studentId, psychologistId } = req.body;
    if (!reportHash || !studentId || !psychologistId) return res.status(400).json({ success: false, error: 'Missing required fields' });

    const blockData = {
      type: 'PERMISSION_REVOKE',
      reportHash,
      studentId,
      psychologistId,
      timestamp: new Date().toISOString()
    };

    const newBlock = blockchain.addBlock(blockData);
    console.log(`❌ Permission revoked on blockchain: Block #${newBlock.index}`);

    res.json({ success: true, transactionHash: newBlock.hash, blockIndex: newBlock.index });
  } catch (error) {
    console.error('Permission revoke error:', error);
    res.status(500).json({ success: false, error: 'Failed to revoke permission' });
  }
});

// Get blockchain info
app.get('/api/chain', (req, res) => {
  res.json({ chain: blockchain.chain, length: blockchain.chain.length, valid: blockchain.validateChain() });
});

// Get specific block
app.get('/api/block/:hash', (req, res) => {
  const { hash } = req.params;
  const block = blockchain.getBlockByHash(hash);
  if (!block) return res.status(404).json({ error: 'Block not found' });
  res.json(block);
});

// Verify report integrity
app.post('/api/verify', (req, res) => {
  try {
    const { reportData, transactionHash } = req.body;
    const block = blockchain.getBlockByHash(transactionHash);
    if (!block) return res.json({ valid: false, error: 'Block not found' });

    const reportHash = crypto.createHash('sha256').update(JSON.stringify(reportData)).digest('hex');
    const storedHash = block.data.reportHash;

    res.json({ valid: reportHash === storedHash, blockIndex: block.index, timestamp: block.timestamp, reportHash, storedHash });
  } catch (error) {
    console.error('Verification error:', error);
    res.status(500).json({ valid: false, error: 'Verification failed' });
  }
});

// ---------------------------
// Start server
// ---------------------------
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`⛓️  Blockchain service running on port ${PORT}`);
  console.log(`📦 Genesis block hash: ${blockchain.chain[0].hash}`);
});
