const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type ApiResult<T> = { data?: T; error?: string };

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<ApiResult<T>> {
  try {
    const token =
      typeof window !== "undefined" ? localStorage.getItem("token") : null;
    
    const headers: Record<string, string> = {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...((options?.headers as Record<string, string>) || {}),
    };

    // Only set Content-Type to application/json if it's not FormData and not already set
    if (!(options?.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }

    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers,
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return { error: body.detail || `Request failed (${res.status})` };
    }

    const data = await res.json();
    return { data };
  } catch (err: any) {
    return { error: err.message || "Network error" };
  }
}

// ──── Auth API ────
export async function register(payload: {
  username: string;
  email: string;
  password: string;
}) {
  return apiFetch<{ token: string; user: any }>("/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function login(payload: { email: string; password: string }) {
  const result = await apiFetch<{ token: string; user: any }>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  if (result.data?.token) {
    localStorage.setItem("token", result.data.token);
  }

  return result;
}

export async function getCurrentUser() {
  return apiFetch<{ user: any }>("/auth/me");
}

export async function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user");
  localStorage.removeItem("isAuthenticated");
}
