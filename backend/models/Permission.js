const mongoose = require('mongoose');

const permissionSchema = new mongoose.Schema({
  studentId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  psychologistId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  reportId: { type: mongoose.Schema.Types.ObjectId, ref: 'Report', required: true },
  status: { type: String, enum: ['active', 'revoked', 'expired'], default: 'active' },
  expiresAt: { type: Date, required: true },
  accessedAt: { type: Date, default: null }
}, {
  timestamps: true
});

// Index for efficient querying
permissionSchema.index({ studentId: 1, psychologistId: 1, reportId: 1 });
permissionSchema.index({ expiresAt: 1 });

// Virtual to check if expired dynamically
permissionSchema.virtual('isExpired').get(function() {
  return this.status === 'active' && new Date() > this.expiresAt;
});

module.exports = mongoose.model('Permission', permissionSchema);
