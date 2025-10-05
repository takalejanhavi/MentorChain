const request = require('supertest');
const mongoose = require('mongoose');
const app = require('../backend/server');
const User = require('../backend/models/User');
const Assessment = require('../backend/models/Assessment');
const { calculateTraitScores } = require('./services/assessmentService');

describe('Psychometric Assessment API', () => {
  let authToken;
  let userId;

  beforeAll(async () => {
    // Connect to test database
    await mongoose.connect(process.env.MONGODB_TEST_URI || 'mongodb://localhost:27017/careerguide_test');
    
    // Create test user
    const testUser = new User({
      name: 'Test Student',
      email: 'test@example.com',
      password: 'password123',
      role: 'student'
    });
    await testUser.save();
    userId = testUser._id;

    // Get auth token
    const loginResponse = await request(app)
      .post('/api/auth/login')
      .send({
        email: 'test@example.com',
        password: 'password123'
      });
    
    authToken = loginResponse.body.token;
  });

  afterAll(async () => {
    // Clean up test data
    await User.deleteMany({});
    await Assessment.deleteMany({});
    await mongoose.connection.close();
  });

  describe('POST /api/assessment/submit', () => {
    it('should submit assessment successfully', async () => {
      const assessmentData = {
        session_id: 'test_session_123',
        responses: generateTestResponses(),
        duration_seconds: 480,
        device_type: 'desktop',
        timestamp: new Date().toISOString()
      };

      const response = await request(app)
        .post('/api/assessment/submit')
        .set('Authorization', `Bearer ${authToken}`)
        .send(assessmentData);

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('assessment_id');
      expect(response.body).toHaveProperty('predictions');
      expect(response.body).toHaveProperty('trait_scores');
      expect(response.body.predictions).toHaveLength(3);
    });

    it('should reject invalid responses', async () => {
      const invalidData = {
        session_id: 'test_session_invalid',
        responses: [], // Empty responses
        duration_seconds: 480,
        device_type: 'desktop'
      };

      const response = await request(app)
        .post('/api/assessment/submit')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidData);

      expect(response.status).toBe(400);
      expect(response.body.message).toContain('Invalid responses');
    });

    it('should require authentication', async () => {
      const response = await request(app)
        .post('/api/assessment/submit')
        .send({});

      expect(response.status).toBe(401);
    });
  });

  describe('GET /api/assessment/:id', () => {
    let assessmentId;

    beforeEach(async () => {
      // Create test assessment
      const assessment = new Assessment({
        userId,
        sessionId: 'test_session_get',
        responses: generateTestResponses(),
        traitScores: {
          big_five: {
            Openness: 7.5,
            Conscientiousness: 8.2,
            Extraversion: 6.1,
            Agreeableness: 7.8,
            Neuroticism: 3.4
          },
          riasec: {
            Realistic: 5.2,
            Investigative: 8.7,
            Artistic: 6.3,
            Social: 7.1,
            Enterprising: 5.9,
            Conventional: 6.8
          }
        },
        predictions: [
          { career: 'IT Professional → Data Science', probability: 0.85 },
          { career: 'Engineer → Computer Science', probability: 0.72 },
          { career: 'Scientist → Physics', probability: 0.68 }
        ],
        modelVersion: 'v2025.01.27',
        deviceType: 'desktop',
        durationSeconds: 480,
        status: 'completed'
      });
      
      await assessment.save();
      assessmentId = assessment._id;
    });

    it('should retrieve assessment successfully', async () => {
      const response = await request(app)
        .get(`/api/assessment/${assessmentId}`)
        .set('Authorization', `Bearer ${authToken}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('_id');
      expect(response.body).toHaveProperty('traitScores');
      expect(response.body).toHaveProperty('predictions');
      expect(response.body.userId.toString()).toBe(userId.toString());
    });

    it('should not allow access to other users assessments', async () => {
      // Create another user
      const otherUser = new User({
        name: 'Other User',
        email: 'other@example.com',
        password: 'password123',
        role: 'student'
      });
      await otherUser.save();

      // Get auth token for other user
      const loginResponse = await request(app)
        .post('/api/auth/login')
        .send({
          email: 'other@example.com',
          password: 'password123'
        });

      const otherToken = loginResponse.body.token;

      const response = await request(app)
        .get(`/api/assessment/${assessmentId}`)
        .set('Authorization', `Bearer ${otherToken}`);

      expect(response.status).toBe(404);
    });
  });

  describe('Trait Calculation', () => {
    it('should calculate Big Five traits correctly', () => {
      const responses = generateTestResponses();
      const traitScores = calculateTraitScores(responses);

      expect(traitScores).toHaveProperty('big_five');
      expect(traitScores).toHaveProperty('riasec');
      
      // Check Big Five traits are in valid range (1-10)
      Object.values(traitScores.big_five).forEach(score => {
        expect(score).toBeGreaterThanOrEqual(1);
        expect(score).toBeLessThanOrEqual(10);
      });

      // Check RIASEC scores are in valid range (1-10)
      Object.values(traitScores.riasec).forEach(score => {
        expect(score).toBeGreaterThanOrEqual(1);
        expect(score).toBeLessThanOrEqual(10);
      });
    });

    it('should handle reverse-scored items correctly', () => {
      // Create responses with all 5s (strongly agree)
      const responses = Array.from({ length: 35 }, (_, i) => ({
        question_id: i + 1,
        response: 5,
        response_time_ms: 3000
      }));

      const traitScores = calculateTraitScores(responses);
      
      // Neuroticism should be lower due to reverse scoring
      expect(traitScores.big_five.Neuroticism).toBeLessThan(7);
    });
  });

  describe('Model Integration', () => {
    it('should handle model service errors gracefully', async () => {
      // Mock model service failure
      const originalEnv = process.env.MODEL_API_URL;
      process.env.MODEL_API_URL = 'http://invalid-url:9999/api';

      const assessmentData = {
        session_id: 'test_session_error',
        responses: generateTestResponses(),
        duration_seconds: 480,
        device_type: 'desktop',
        timestamp: new Date().toISOString()
      };

      const response = await request(app)
        .post('/api/assessment/submit')
        .set('Authorization', `Bearer ${authToken}`)
        .send(assessmentData);

      // Should handle error gracefully
      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('message');

      // Restore original environment
      process.env.MODEL_API_URL = originalEnv;
    });
  });
});

// Helper function to generate test responses
function generateTestResponses() {
  return Array.from({ length: 35 }, (_, i) => ({
    question_id: i + 1,
    response: Math.floor(Math.random() * 5) + 1, // Random 1-5
    response_time_ms: Math.floor(Math.random() * 5000) + 1000 // Random 1-6 seconds
  }));
}