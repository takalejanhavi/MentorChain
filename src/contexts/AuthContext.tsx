import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { api } from '../services/api';

interface User {
  id: string;
  email: string;
  name: string;
  role: 'student' | 'psychologist';
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string, role: 'student' | 'psychologist') => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      fetchUser();
    } else {
      setLoading(false);
    }
  }, []);

  // Fetch current user from backend
  const fetchUser = async () => {
    try {
      const response = await api.get('/auth/me');
      const userData = response.data.user;

      if (!userData || !userData.role) {
        throw new Error('Invalid user data from backend');
      }

      setUser({
        id: userData._id || userData.id,
        email: userData.email,
        name: userData.name,
        role: userData.role,
      });
    } catch (error) {
      console.error('Failed to fetch user:', error);
      logout(); // clear token if invalid
    } finally {
      setLoading(false);
    }
  };

  // Login user
  const login = async (email: string, password: string) => {
    try {
      const response = await api.post('/auth/login', { email, password });
      const { token, user: userData } = response.data;

      if (!token || !userData || !userData.role) {
        throw new Error('Invalid login response from server');
      }

      localStorage.setItem('token', token);
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

      setUser({
        id: userData._id || userData.id,
        email: userData.email,
        name: userData.name,
        role: userData.role,
      });
    } catch (err: any) {
      console.error('Login error:', err.response?.data?.message || err.message);
      throw new Error(err.response?.data?.message || 'Login failed');
    }
  };

  // Register user
  const register = async (name: string, email: string, password: string, role: 'student' | 'psychologist') => {
    try {
      const response = await api.post('/auth/register', { name, email, password, role });
      const { token, user: userData } = response.data;

      if (!token || !userData || !userData.role) {
        throw new Error('Invalid register response from server');
      }

      localStorage.setItem('token', token);
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

      setUser({
        id: userData._id || userData.id,
        email: userData.email,
        name: userData.name,
        role: userData.role,
      });
    } catch (err: any) {
      console.error('Registration error:', err.response?.data?.message || err.message);
      throw new Error(err.response?.data?.message || 'Registration failed');
    }
  };

  // Logout user
  const logout = () => {
    localStorage.removeItem('token');
    delete api.defaults.headers.common['Authorization'];
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
