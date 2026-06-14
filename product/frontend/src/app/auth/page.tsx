"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { register, login } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { Mic, ArrowLeft } from "lucide-react";

export default function AuthPage() {
  const [authMode, setAuthMode] = useState<"select" | "signup" | "login">(
    "select",
  );

  return (
    <main className="relative min-h-screen w-full overflow-hidden bg-linear-to-br from-orange-50 via-white to-blue-50">
      {/* Decorative Blurred Orbs */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full bg-orange-400/20 blur-[120px]" />
        <div className="absolute -bottom-32 -right-32 w-[500px] h-[500px] rounded-full bg-blue-400/20 blur-[120px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-purple-300/10 blur-[100px]" />
      </div>

      {/* Navigation Bar */}
      <nav className="fixed left-0 right-0 top-0 z-50 flex items-center justify-between px-6 py-6 md:px-12">
        <Link
          href="/"
          className="flex items-center gap-2 transition-transform hover:scale-105"
        >
          <div className="p-2 rounded-xl bg-orange-500/10">
            <Mic className="w-6 h-6 text-orange-500" />
          </div>
          <span className="font-bold text-lg text-foreground">VoiceScore</span>
        </Link>
        <Link
          href="/"
          className="font-sans text-sm font-medium text-foreground/60 transition-colors hover:text-foreground"
        >
          Back to Home
        </Link>
      </nav>

      {/* Main Content */}
      <div className="relative z-10 flex min-h-screen items-center justify-center px-6 py-24 md:px-12">
        <div className="w-full max-w-md">
          {authMode === "select" && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="mb-8 text-center">
                <h1 className="mb-3 text-4xl font-light leading-tight text-foreground md:text-5xl">
                  <span className="text-balance">Welcome to VoiceScore</span>
                </h1>
                <p className="text-lg text-foreground/60">
                  Master your pronunciation with AI-powered feedback.
                </p>
              </div>

              <div className="space-y-4">
                <button
                  onClick={() => setAuthMode("signup")}
                  className="w-full px-6 py-4 rounded-2xl font-semibold text-white bg-orange-500 hover:bg-orange-600 shadow-lg shadow-orange-500/20 transition-all hover:-translate-y-0.5 hover:shadow-xl"
                >
                  Sign Up
                </button>
                <button
                  onClick={() => setAuthMode("login")}
                  className="w-full px-6 py-4 rounded-2xl font-semibold text-foreground bg-white border border-border/60 hover:bg-secondary/40 shadow-sm transition-all hover:-translate-y-0.5"
                >
                  Log In
                </button>
              </div>
            </div>
          )}

          {authMode === "signup" && (
            <SignUpForm
              onBack={() => setAuthMode("select")}
              onSwitchToLogin={() => setAuthMode("login")}
            />
          )}

          {authMode === "login" && (
            <LoginForm onBack={() => setAuthMode("select")} />
          )}
        </div>
      </div>
    </main>
  );
}

function SignUpForm({
  onBack,
  onSwitchToLogin,
}: {
  onBack: () => void;
  onSwitchToLogin: () => void;
}) {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) newErrors.name = "Name is required";
    if (!formData.email.trim()) newErrors.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email))
      newErrors.email = "Please enter a valid email";
    if (!formData.password) newErrors.password = "Password is required";
    else if (formData.password.length < 8)
      newErrors.password = "Password must be at least 8 characters";
    if (formData.password !== formData.confirmPassword)
      newErrors.confirmPassword = "Passwords do not match";

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    try {
      const { data: regData, error: regError } = await register({
        username: formData.name,
        email: formData.email,
        password: formData.password,
      });

      if (regError || !regData) {
        setErrors({ form: regError || "Registration failed" });
        setIsLoading(false);
        return;
      }

      // Auto-login after registration
      const { data: loginData, error: loginError } = await login({
        email: formData.email,
        password: formData.password,
      });

      if (loginError || !loginData) {
        onSwitchToLogin();
        setIsLoading(false);
        return;
      }

      localStorage.setItem("user", JSON.stringify(loginData.user));
      localStorage.setItem("isAuthenticated", "true");
      window.location.href = "/dashboard";
    } catch {
      setErrors({ form: "Network error. Please try again." });
    }
    setIsLoading(false);
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <button
          onClick={onBack}
          className="mb-6 text-sm text-foreground/60 hover:text-foreground transition-colors flex items-center gap-1"
        >
          <ArrowLeft size={14} /> Back
        </button>
        <h2 className="mb-3 text-3xl font-light leading-tight text-foreground md:text-4xl">
          Create Your Account
        </h2>
        <p className="text-foreground/60">
          Start your pronunciation journey today.
        </p>
      </div>

      {errors.form && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 text-red-600 text-sm border border-red-200">
          {errors.form}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        {(
          [
            ["name", "Full Name", "text", "John Doe"],
            ["email", "Email Address", "email", "john@example.com"],
            ["password", "Password", "password", "At least 8 characters"],
            [
              "confirmPassword",
              "Confirm Password",
              "password",
              "Re-enter your password",
            ],
          ] as const
        ).map(([key, label, type, placeholder]) => (
          <div key={key}>
            <label className="mb-2 block text-sm font-medium text-foreground">
              {label}
            </label>
            <input
              type={type}
              value={formData[key]}
              onChange={(e) =>
                setFormData({ ...formData, [key]: e.target.value })
              }
              className={`w-full rounded-xl border bg-white/60 px-4 py-3 text-foreground backdrop-blur-sm transition-all placeholder:text-foreground/30 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 ${
                errors[key] ? "border-red-400" : "border-border/40"
              }`}
              placeholder={placeholder}
              disabled={isLoading}
            />
            {errors[key] && (
              <p className="mt-1 text-sm text-red-500">{errors[key]}</p>
            )}
          </div>
        ))}

        <button
          type="submit"
          disabled={isLoading}
          className="w-full px-6 py-4 rounded-2xl font-semibold text-white bg-orange-500 hover:bg-orange-600 shadow-lg shadow-orange-500/20 transition-all hover:-translate-y-0.5 disabled:opacity-50"
        >
          {isLoading ? "Creating Account..." : "Create Account"}
        </button>
      </form>
    </div>
  );
}

function LoginForm({ onBack }: { onBack: () => void }) {
  const router = useRouter();
  const { refreshUser } = useAuth();
  const [formData, setFormData] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Record<string, string> = {};

    if (!formData.email.trim()) newErrors.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email))
      newErrors.email = "Please enter a valid email";
    if (!formData.password) newErrors.password = "Password is required";

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);
    try {
      const { data, error } = await login({
        email: formData.email,
        password: formData.password,
      });

      if (error || !data) {
        setErrors({ form: error || "Login failed" });
        setIsLoading(false);
        return;
      }

      localStorage.setItem("user", JSON.stringify(data.user));
      localStorage.setItem("isAuthenticated", "true");

      window.location.href = "/dashboard";
    } catch {
      setErrors({ form: "Network error. Please try again." });
    }
    setIsLoading(false);
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <button
          onClick={onBack}
          className="mb-6 text-sm text-foreground/60 hover:text-foreground transition-colors flex items-center gap-1"
        >
          <ArrowLeft size={14} /> Back
        </button>
        <h2 className="mb-3 text-3xl font-light leading-tight text-foreground md:text-4xl">
          Welcome Back
        </h2>
        <p className="text-foreground/60">
          Log in to access your pronunciation dashboard.
        </p>
      </div>

      {errors.form && (
        <div className="mb-4 p-3 rounded-lg bg-red-50 text-red-600 text-sm border border-red-200">
          {errors.form}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label className="mb-2 block text-sm font-medium text-foreground">
            Email Address
          </label>
          <input
            type="email"
            value={formData.email}
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
            className={`w-full rounded-xl border bg-white/60 px-4 py-3 text-foreground backdrop-blur-sm transition-all placeholder:text-foreground/30 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 ${
              errors.email ? "border-red-400" : "border-border/40"
            }`}
            placeholder="john@example.com"
            disabled={isLoading}
          />
          {errors.email && (
            <p className="mt-1 text-sm text-red-500">{errors.email}</p>
          )}
        </div>

        <div>
          <label className="mb-2 block text-sm font-medium text-foreground">
            Password
          </label>
          <input
            type="password"
            value={formData.password}
            onChange={(e) =>
              setFormData({ ...formData, password: e.target.value })
            }
            className={`w-full rounded-xl border bg-white/60 px-4 py-3 text-foreground backdrop-blur-sm transition-all placeholder:text-foreground/30 focus:border-orange-500 focus:outline-none focus:ring-2 focus:ring-orange-500/20 ${
              errors.password ? "border-red-400" : "border-border/40"
            }`}
            placeholder="Enter your password"
            disabled={isLoading}
          />
          {errors.password && (
            <p className="mt-1 text-sm text-red-500">{errors.password}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full px-6 py-4 rounded-2xl font-semibold text-white bg-orange-500 hover:bg-orange-600 shadow-lg shadow-orange-500/20 transition-all hover:-translate-y-0.5 disabled:opacity-50"
        >
          {isLoading ? "Logging in..." : "Log In"}
        </button>
      </form>
    </div>
  );
}
