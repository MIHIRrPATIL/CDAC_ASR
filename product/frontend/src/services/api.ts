import axios from "axios";

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
  };
  analysis: {
    error_stats: { sub: number; ins: number; del: number };
    aligned_pairs: [string, string][];
  };
  feedback: string | null;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const analyzeAudio = async (
  file: File,
  targetWord?: string,
): Promise<PronunciationResponse> => {
  const formData = new FormData();
  formData.append("audio_file", file);

  if (targetWord) {
    formData.append("target_word", targetWord);
  }

  const token = typeof window !== 'undefined' ? localStorage.getItem("token") : null;
  const response = await axios.post<PronunciationResponse>(
    `${API_URL}/analyze`,
    formData,
    {
      headers: { 
        "Content-Type": "multipart/form-data",
        ...(token ? { "Authorization": `Bearer ${token}` } : {})
      },
    },
  );

  return response.data;
};
