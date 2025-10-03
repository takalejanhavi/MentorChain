const FormData = require('form-data');
const axios = require('axios');

const WEB3_STORAGE_TOKEN = process.env.WEB3STORAGE_TOKEN;
const WEB3_STORAGE_API = 'https://api.web3.storage';

/**
 * Upload encrypted report to IPFS via Web3.Storage
 * @param {object} encryptedReport - Encrypted report data
 * @param {string} filename - Filename for the report
 * @returns {Promise<string>} - IPFS CID
 */
async function uploadToIPFS(encryptedReport, filename = 'career-report.json') {
  try {
    if (!WEB3_STORAGE_TOKEN) {
      throw new Error('Web3.Storage token not configured');
    }

    // Create form data
    const formData = new FormData();
    const reportBuffer = Buffer.from(JSON.stringify(encryptedReport), 'utf8');
    formData.append('file', reportBuffer, {
      filename,
      contentType: 'application/json'
    });

    // Upload to Web3.Storage
    const response = await axios.post(`${WEB3_STORAGE_API}/upload`, formData, {
      headers: {
        'Authorization': `Bearer ${WEB3_STORAGE_TOKEN}`,
        ...formData.getHeaders()
      },
      timeout: 30000
    });

    if (response.data && response.data.cid) {
      console.log(`📦 Report uploaded to IPFS: ${response.data.cid}`);
      return response.data.cid;
    } else {
      throw new Error('Invalid response from Web3.Storage');
    }

  } catch (error) {
    console.error('IPFS upload error:', error.message);
    throw new Error(`IPFS upload failed: ${error.message}`);
  }
}

/**
 * Retrieve encrypted report from IPFS
 * @param {string} cid - IPFS CID
 * @returns {Promise<object>} - Encrypted report data
 */
async function retrieveFromIPFS(cid) {
  try {
    const response = await axios.get(`https://${cid}.ipfs.w3s.link/career-report.json`, {
      timeout: 15000
    });

    return response.data;
  } catch (error) {
    console.error('IPFS retrieval error:', error.message);
    throw new Error(`IPFS retrieval failed: ${error.message}`);
  }
}

/**
 * Check if CID is valid and accessible
 * @param {string} cid - IPFS CID to verify
 * @returns {Promise<boolean>} - True if accessible
 */
async function verifyCID(cid) {
  try {
    const response = await axios.head(`https://${cid}.ipfs.w3s.link/career-report.json`, {
      timeout: 10000
    });
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

module.exports = {
  uploadToIPFS,
  retrieveFromIPFS,
  verifyCID
};