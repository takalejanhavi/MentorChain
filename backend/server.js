const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
require('dotenv').config();

const authRoutes = require('./routes/auth');
const dashboardRoutes = require('./routes/dashboard');
const testRoutes = require('./routes/tests');
const reportRoutes = require('./routes/reports');
const permissionRoutes = require('./routes/permissions');
const User = require('./models/User'); 

const app = express();


// Middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb+srv://janhavit:database@mernecom.igrid.mongodb.net/career-guidance?retryWrites=true&w=majority', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log('✅ Connected to MongoDB'))
.catch(err => console.error('❌ MongoDB connection error:', err));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/tests', testRoutes);
app.use('/api/reports', reportRoutes);
app.use('/api/permissions', permissionRoutes);

app.post('/api/test-save', async (req, res) => {
  try {
    const testUser = new User({
      name: 'Test User',
      email: `test${Date.now()}@example.com`,
      password: '123456',
      role: 'student'
    });
    await testUser.save();
    console.log('Saved test user:', testUser);
    res.json({ message: 'User saved', user: testUser });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Failed to save test user', error: err.message });
  }
});


// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Career Guidance API is running' });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});