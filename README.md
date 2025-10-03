# Career Guidance Platform

A comprehensive, production-ready career guidance platform built with MERN stack, Python machine learning, and blockchain technology.

## 🚀 Features

### For Students
- **AI-Powered Career Assessment**: Take comprehensive personality and academic tests
- **Personalized Reports**: Get detailed career recommendations with visualizations
- **Blockchain Security**: All reports are securely stored on blockchain
- **Permission Management**: Control who can access your reports
- **Professional Dashboard**: Track your progress and assessments

### For Psychologists
- **Student Report Access**: View reports that students have granted access to
- **Professional Analytics**: Analyze student career patterns and recommendations
- **Secure Permissions**: Access reports only with proper student consent
- **Comprehensive Insights**: Detailed personality and academic analysis

### Technical Features
- **FAANG-Level UI/UX**: Modern, responsive design with web3 aesthetics
- **Advanced ML Model**: Python-based career prediction using ensemble methods
- **Blockchain Integration**: Secure report storage and permission management
- **Microservices Architecture**: Scalable backend with separate services
- **Production Ready**: Dockerized deployment with comprehensive documentation

## 🏗️ Architecture

```
Frontend (React + Vite) ←→ Backend (Node.js + Express) ←→ MongoDB Atlas
                    ↕                    ↕
            Python ML Service    Blockchain Service
```

## 📋 Prerequisites

- Node.js (v18+)
- Python (v3.11+)
- MongoDB Atlas account
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd career-guidance-platform
```

### 2. Frontend Setup
```bash
npm install
```

### 3. Backend Setup
```bash
cd backend
npm install
```

### 4. Python Service Setup
```bash
cd ../python-service
pip install -r requirements.txt
```

### 5. Blockchain Service Setup
```bash
cd ../blockchain-service
npm install
```

### 6. Environment Configuration

Copy `.env.example` to `.env` and configure:

```env
# MongoDB Atlas (Required)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/careerguide

# JWT Secret (Required)
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters-long

# Service URLs
VITE_API_URL=http://localhost:5000/api
VITE_MODEL_URL=http://localhost:5002/api
VITE_BLOCKCHAIN_URL=http://localhost:5001/api
```

## 🚀 Running the Application

### Development Mode

1. **Start Backend Services** (in separate terminals):
```bash
# Backend API
cd backend && npm run dev

# Python ML Service
cd python-service && python app.py

# Blockchain Service
cd blockchain-service && npm run dev
```

2. **Start Frontend**:
```bash
npm run dev
```

3. **Access Application**:
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000
- Python Service: http://localhost:5002
- Blockchain Service: http://localhost:5001

### Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

## 📊 Machine Learning Model

The Python service includes a trained career prediction model that:

- **Analyzes 10 key factors**: Academic scores, personality traits, and extracurricular involvement
- **Predicts top 3 careers** with confidence percentages
- **Uses ensemble methods**: Random Forest with advanced preprocessing
- **Generates visualizations**: Professional charts and reports
- **Provides recommendations**: Personalized career guidance

### Model Features:
- Math Score, Science Score, English Score
- Personality traits (Big Five model)
- Extracurricular involvement
- Overall academic percentage

### Supported Careers:
Doctor, Engineer, IT Professional, Designer, Business/Entrepreneur, Teacher, Psychologist, Scientist, Lawyer, Accountant, Artist, Pilot, Musician

## 🔒 Blockchain Security

### Features:
- **Secure Report Storage**: All career reports are hashed and stored on blockchain
- **Permission Management**: Granular access control with expiration dates
- **Immutable Records**: Tamper-proof audit trail of all transactions
- **Verification System**: Integrity checking for all stored reports

### Blockchain Operations:
- Store report hashes securely
- Grant/revoke permissions with timestamps
- Verify report integrity
- Audit access logs

## 🗄️ Database Schema

### Users Collection
```javascript
{
  name: String,
  email: String (unique),
  password: String (hashed),
  role: 'student' | 'psychologist',
  isActive: Boolean,
  timestamps
}
```

### Reports Collection
```javascript
{
  userId: ObjectId,
  answers: {
    Math_Score: Number,
    Science_Score: Number,
    // ... other assessment fields
  },
  report: {
    top3_careers: Map,
    accuracy: Number,
    recommendations: String
  },
  blockchainHash: String,
  timestamps
}
```

### Permissions Collection
```javascript
{
  studentId: ObjectId,
  psychologistId: ObjectId,
  reportId: ObjectId,
  status: 'active' | 'revoked' | 'expired',
  expiresAt: Date,
  timestamps
}
```

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/me` - Get current user

### Dashboard
- `GET /api/dashboard` - Get dashboard stats and activity

### Tests & Reports
- `POST /api/tests/submit` - Submit career assessment
- `GET /api/reports` - Get user reports
- `GET /api/reports/:id` - Get specific report

### Permissions
- `POST /api/permissions/grant` - Grant report access
- `GET /api/permissions` - Get user permissions
- `DELETE /api/permissions/:id` - Revoke permission

### Machine Learning
- `POST /api/predict` - Generate career predictions
- `GET /api/health` - Model service health check

### Blockchain
- `POST /api/store` - Store report on blockchain
- `POST /api/grant-permission` - Blockchain permission grant
- `POST /api/verify` - Verify report integrity

## 🎨 UI/UX Features

### Design System
- **Color Palette**: Professional blue/purple gradients with proper contrast
- **Typography**: Clean, readable fonts with proper hierarchy
- **Spacing**: Consistent 8px grid system
- **Components**: Reusable, accessible UI components

### Responsive Design
- Mobile-first approach
- Tablet and desktop optimizations
- Touch-friendly interactions
- Progressive enhancement

### Animations
- Smooth page transitions
- Micro-interactions for better UX
- Loading states and feedback
- Framer Motion integration

## 🔧 Configuration

### MongoDB Atlas Setup
1. Create MongoDB Atlas account
2. Create new cluster
3. Add database user
4. Whitelist IP addresses
5. Get connection string

### Environment Variables
All environment variables are documented in `.env.example`

### Production Deployment
- Use strong JWT secrets
- Enable HTTPS
- Set up proper CORS policies
- Configure MongoDB Atlas security
- Set up monitoring and logging

## 📱 Mobile Responsiveness

The application is fully responsive with:
- Mobile-optimized navigation
- Touch-friendly interactions
- Responsive charts and visualizations
- Adaptive layouts for all screen sizes

## 🧪 Testing

### Frontend Testing
```bash
npm run test
```

### Backend Testing
```bash
cd backend && npm run test
```

### API Testing
Use the included Postman collection for comprehensive API testing.

## 🚀 Deployment

### Cloud Deployment Options

1. **Frontend**: Vercel, Netlify, AWS S3
2. **Backend**: Railway, Render, DigitalOcean
3. **Python Service**: Railway, Heroku, AWS Lambda
4. **Database**: MongoDB Atlas (already cloud-based)

### Docker Production
```bash
# Build for production
docker-compose -f docker-compose.prod.yml up --build

# Scale services
docker-compose up --scale backend=3 --scale python-service=2
```

## 📖 Documentation

### API Documentation
- Comprehensive API documentation available in `/docs`
- Postman collection for testing
- OpenAPI/Swagger specs

### Development Guide
- Code style guidelines
- Contributing instructions
- Architecture decisions

## 🔐 Security Features

- **Authentication**: JWT-based with secure password hashing
- **Authorization**: Role-based access control
- **Data Privacy**: Blockchain-secured report storage
- **Input Validation**: Comprehensive input sanitization
- **CORS**: Properly configured cross-origin requests

## 🎯 Future Enhancements

1. **Advanced ML Models**: Deep learning integration
2. **Real-time Features**: WebSocket-based notifications
3. **Mobile App**: React Native application
4. **Advanced Analytics**: Comprehensive reporting dashboard
5. **Third-party Integrations**: University/job portal connections

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create GitHub issues for bugs
- Check documentation for common questions
- Review API documentation for integration help

---

