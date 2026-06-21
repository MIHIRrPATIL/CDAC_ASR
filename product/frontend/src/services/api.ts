import axios from "axios";

export interface GopDetail {
  phoneme: string;
  start_ms: number;
  end_ms: number;
  gop_prob: number;
  is_correct: boolean;
}

export interface WordAnalysisItem {
  word: string;
  phonemes: string[];
  aligned_pairs: [string, string][];
  accuracy: number;
  error_stats: { sub: number; ins: number; del: number };
  status: "correct" | "incorrect";
}

export interface PronunciationResponse {
  scores: {
    phoneme: number;
    duration: { accuracy: number; avg_ratio: number; error_ms: number };
    pitch: { similarity: number; error_hz: number; correlation: number };
    stress: {
      accuracy: number;
      error_stats: {
        missing_stress: number;
        extra_stress: number;
        wrong_stress: number;
      };
    };
    gop_details?: GopDetail[];
  };
  analysis: {
    error_stats: { sub: number; ins: number; del: number };
    aligned_pairs: [string, string][];
    words_analysis?: WordAnalysisItem[];
  };
  feedback: string[] | null;
  target_word?: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 
  (typeof window !== "undefined" ? window.location.origin : "http://localhost:8000");

// Headers generator
const getAuthHeaders = () => {
  const token = typeof window !== 'undefined' ? localStorage.getItem("token") : null;
  return token ? { "Authorization": `Bearer ${token}` } : {};
};

export const analyzeAudio = async (
  file: File,
  targetWord?: string,
): Promise<PronunciationResponse> => {
  const formData = new FormData();
  formData.append("audio_file", file);

  if (targetWord) {
    formData.append("target_word", targetWord);
  }

  const response = await axios.post<PronunciationResponse>(
    `${API_URL}/analyze`,
    formData,
    {
      headers: { 
        "Content-Type": "multipart/form-data",
        ...getAuthHeaders()
      },
    },
  );

  return response.data;
};

// ──── New Interfaces ────
export interface HeatmapItem {
  phoneme: string;
  accuracy: number;
  total_practiced: number;
}

export interface ScoreTrendItem {
  date: string;
  score: number;
}

export interface DashboardStats {
  overall_accuracy: number;
  practice_seconds: number;
  daily_streak: number;
  global_rank: number;
  history: ScoreTrendItem[];
  heatmap: HeatmapItem[];
}

export interface SpacedRepetitionItem {
  id: string;
  word: string;
  phonemes: string;
  easeFactor: number;
  interval: number;
  repetitions: number;
  nextReviewAt: string;
}

export interface CustomWordList {
  id: string;
  title: string;
  description?: string;
  entries: WordListEntry[];
}

export interface WordListEntry {
  id: string;
  listId: string;
  word: string;
  phonemes?: string;
}

export interface AITextResponse {
  paragraph: string;
  targeted_phonemes: string[];
}

export interface RoleplayResponse {
  response: string;
  corrections: string;
  suggested_replies?: string[];
}

export interface AIPair {
  word1: string;
  word2: string;
}

export interface AIDrillResponse {
  label: string;
  description: string;
  pairs: AIPair[];
}

// ──── New API Services ────
export const getTTSAudioUrl = (text: string, slow: boolean = false): string => {
  return `${API_URL}/tts/generate?text=${encodeURIComponent(text)}&slow=${slow}`;
};

export const getDashboardStats = async (): Promise<DashboardStats> => {
  const response = await axios.get<DashboardStats>(
    `${API_URL}/dashboard/stats`,
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const getSRQueue = async (): Promise<SpacedRepetitionItem[]> => {
  const response = await axios.get<SpacedRepetitionItem[]>(
    `${API_URL}/spaced-repetition/queue`,
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const addSRCard = async (word: string, phonemes: string): Promise<SpacedRepetitionItem> => {
  const response = await axios.post<SpacedRepetitionItem>(
    `${API_URL}/spaced-repetition/add`,
    { word, phonemes },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const reviewSRCard = async (sr_id: string, overall_score: number): Promise<SpacedRepetitionItem> => {
  const response = await axios.post<SpacedRepetitionItem>(
    `${API_URL}/spaced-repetition/review`,
    { sr_id, overall_score },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const getWordLists = async (): Promise<CustomWordList[]> => {
  const response = await axios.get<CustomWordList[]>(
    `${API_URL}/lists`,
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const createWordList = async (title: string, description?: string): Promise<CustomWordList> => {
  const response = await axios.post<CustomWordList>(
    `${API_URL}/lists`,
    { title, description },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const deleteWordList = async (list_id: string): Promise<any> => {
  const response = await axios.delete(
    `${API_URL}/lists/${list_id}`,
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const addListEntry = async (list_id: string, word: string, phonemes?: string): Promise<WordListEntry> => {
  const response = await axios.post<WordListEntry>(
    `${API_URL}/lists/${list_id}/entries`,
    { word, phonemes },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const generateAIText = async (topic?: string): Promise<AITextResponse> => {
  const response = await axios.post<AITextResponse>(
    `${API_URL}/ai/generate-text`,
    { topic },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const postRoleplay = async (dialogue_history: any[], scenario?: string): Promise<RoleplayResponse> => {
  const response = await axios.post<RoleplayResponse>(
    `${API_URL}/ai/roleplay`,
    { dialogue_history, scenario },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const startRoleplay = async (scenario?: string): Promise<RoleplayResponse> => {
  const response = await axios.post<RoleplayResponse>(
    `${API_URL}/ai/start-roleplay`,
    { scenario },
    { headers: getAuthHeaders() }
  );
  return response.data;
};

export const generateAIDrills = async (prompt?: string): Promise<AIDrillResponse> => {
  const response = await axios.post<AIDrillResponse>(
    `${API_URL}/ai/generate-drills`,
    { prompt },
    { headers: getAuthHeaders() }
  );
  return response.data;
};
