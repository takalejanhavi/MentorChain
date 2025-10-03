const crypto = require('crypto');

// AES-256-GCM encryption
const ALGORITHM = 'aes-256-gcm';
const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY || crypto.randomBytes(32);

/**
 * Encrypt data using AES-256-GCM
 * @param {string} text - Text to encrypt
 * @returns {object} - Encrypted data with IV and auth tag
 */
function encrypt(text) {
  try {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(ALGORITHM, ENCRYPTION_KEY);
    cipher.setAAD(Buffer.from('career-report', 'utf8'));
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  } catch (error) {
    console.error('Encryption error:', error);
    throw new Error('Encryption failed');
  }
}

/**
 * Decrypt data using AES-256-GCM
 * @param {object} encryptedData - Object containing encrypted data, IV, and auth tag
 * @returns {string} - Decrypted text
 */
function decrypt(encryptedData) {
  try {
    const { encrypted, iv, authTag } = encryptedData;
    
    const decipher = crypto.createDecipher(ALGORITHM, ENCRYPTION_KEY);
    decipher.setAAD(Buffer.from('career-report', 'utf8'));
    decipher.setAuthTag(Buffer.from(authTag, 'hex'));
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  } catch (error) {
    console.error('Decryption error:', error);
    throw new Error('Decryption failed');
  }
}

/**
 * Generate hash of data for blockchain storage
 * @param {string} data - Data to hash
 * @returns {string} - SHA-256 hash
 */
function generateHash(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

module.exports = {
  encrypt,
  decrypt,
  generateHash
};