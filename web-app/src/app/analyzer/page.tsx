"use client";

import React, { useState } from "react";
import AudioInput from "../../components/AudioInput";
import ResultsDashboard from "../../components/ResultsDashboard";
import { analyzeAudio, PronunciationResponse } from "../../services/api";
import { Sparkles, Brain, Award, Activity } from "lucide-react";
import { CustomCursor } from "../../components/custom-cursor";
import { ProtectedRoute } from "../../components/protected-route";

export default function AnalyzerPage() {
  const [data, setData] = useState<PronunciationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAudioSubmit = async (file: File, targetWord: string) => {
    setIsLoading(true);
    setError(null);
    setData(null);

    try {
      const result = await analyzeAudio(file, targetWord);
      // Ensure metrics exist to prevent crashing ResultsDashboard
      if (result.scores) {
        if (!result.scores.pitch)
          result.scores.pitch = { similarity: 0, error_hz: 0, correlation: 0 };
        if (!result.scores.duration)
          result.scores.duration = { accuracy: 0, avg_ratio: 0, error_ms: 0 };
        if (!result.scores.stress)
          result.scores.stress = {
            accuracy: 0,
            error_stats: {
              missing_stress: 0,
              extra_stress: 0,
              wrong_stress: 0,
            },
          };
      }
      setData(result);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || "Failed to contact the analyzer.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background pt-28 pb-20">
        <CustomCursor />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header Section (NitiAI Dashboard Style) */}
          <section className="mb-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-pretty mb-2 text-foreground">
                  Pronunciation Lab
                </h1>
                <p className="text-muted-foreground font-medium flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-primary" />
                  Record or upload audio to receive clinical-grade phoneme
                  analysis
                </p>
              </div>
            </div>

            {error && (
              <div className="mb-6 p-4 rounded-lg bg-destructive/10 text-destructive flex items-center gap-2">
                <Activity className="w-5 h-5" />
                <span>{error}</span>
              </div>
            )}
          </section>

          {/* Input Section */}
          <section className="mb-10 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-100">
            <div className="p-6 bg-card/60 backdrop-blur-sm rounded-2xl border border-border/10 shadow-lg">
              <AudioInput
                onAudioSubmit={handleAudioSubmit}
                isLoading={isLoading}
              />
            </div>
          </section>

          {/* Results Presentation */}
          {data && (
            <section className="animate-in fade-in slide-in-from-bottom-4 duration-500 delay-200">
              <ResultsDashboard data={data} />
            </section>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}
