"use client";

import * as React from "react";
import { getCurrentUser, logout as apiLogout } from "./api";

type User = {
  id: string;
  name: string;
  email: string;
};

type AuthContextType = {
  user: User | null;
  loading: boolean;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
};

const AuthContext = React.createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUserState] = React.useState<User | null>(null);
  const [loading, setLoading] = React.useState(true);

  const refreshUser = React.useCallback(async () => {
    const token = localStorage.getItem("token");

    if (!token) {
      setUserState(null);
      setLoading(false);
      return;
    }

    try {
      const result = await getCurrentUser();

      if (result.data?.user) {
        const apiUser = result.data.user;
        setUserState({
          id: apiUser.id,
          name: apiUser.name || "",
          email: apiUser.email,
        });
      } else {
        localStorage.removeItem("token");
        setUserState(null);
      }
    } catch (e) {
      console.error("Error fetching user:", e);
      localStorage.removeItem("token");
      setUserState(null);
    }

    setLoading(false);
  }, []);

  React.useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const setUser = React.useCallback((user: User | null) => {
    setUserState(user);
    if (!user) {
      localStorage.removeItem("token");
    }
  }, []);

  const logout = React.useCallback(async () => {
    await apiLogout();
    setUserState(null);
  }, []);

  const isAuthenticated = !!user;

  return (
    <AuthContext.Provider
      value={{ user, loading, isAuthenticated, setUser, logout, refreshUser }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const context = React.useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
