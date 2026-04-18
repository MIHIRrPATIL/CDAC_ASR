"use client";

import React, { useRef, useState, useEffect } from "react";
import {
  Mic,
  Upload,
  Square,
  Loader2,
  Sparkles,
  AudioLines,
} from "lucide-react";

interface AudioInputProps {
  onAudioSubmit: (file: File, targetWord: string) => void;
  isLoading: boolean;
}

export default function AudioInput({
  onAudioSubmit,
  isLoading,
}: AudioInputProps) {
  const [word, setWord] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else {
      setRecordingTime(0);
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && word) {
      onAudioSubmit(file, word);
    }
  };

  const toggleRecording = async () => {
    if (!word) return;

    if (isRecording) {
      mediaRecorder.current?.stop();
      setIsRecording(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });
        mediaRecorder.current = new MediaRecorder(stream);
        audioChunks.current = [];

        mediaRecorder.current.ondataavailable = (event) => {
          if (event.data.size > 0) audioChunks.current.push(event.data);
        };

        mediaRecorder.current.onstop = () => {
          const audioBlob = new Blob(audioChunks.current, {
            type: "audio/wav",
          });
          const file = new File([audioBlob], "recording.wav", {
            type: "audio/wav",
          });
          onAudioSubmit(file, word);
          stream.getTracks().forEach((track) => track.stop());
        };

        mediaRecorder.current.start();
        setIsRecording(true);
      } catch {
        console.error("Microphone access denied.");
      }
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center gap-2 mb-2">
        <div className="p-2 rounded-lg bg-orange-500/10 dark:bg-orange-950/30">
          <Sparkles className="w-5 h-5 text-orange-600 dark:text-orange-500" />
        </div>
        <h2 className="text-xl font-bold text-foreground">Acoustic Input</h2>
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-end">
        <div className="w-full md:w-1/2">
          <label className="block text-sm font-semibold tracking-wider text-muted-foreground uppercase mb-2">
            Target Word / Phrase
          </label>
          <input
            type="text"
            value={word}
            onChange={(e) => setWord(e.target.value)}
            placeholder="e.g., because, temperature"
            className="w-full px-4 py-3 bg-white dark:bg-secondary/20 shadow-sm border border-border/60 dark:border-border/40 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500/50 text-foreground transition-all"
          />
        </div>

        <div className="flex flex-col gap-3 w-full md:w-1/2">
          <div className="flex flex-col xl:flex-row gap-3">
            <button
              onClick={toggleRecording}
              disabled={isLoading || !word}
              className={`flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${
                isRecording
                  ? "bg-red-50 text-red-600 ring-1 ring-red-200 shadow-sm hover:bg-red-100 dark:bg-red-950/40 dark:ring-red-900/50 dark:text-red-400"
                  : "bg-orange-500 hover:bg-orange-600 text-white shadow-md shadow-orange-500/10 dark:shadow-none hover:-translate-y-0.5"
              } disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isRecording ? <Square size={18} /> : <Mic size={18} />}
              {isRecording ? "Stop" : "Record"}
            </button>

            <div className="relative flex-1">
              <input
                type="file"
                accept="audio/wav,audio/*"
                onChange={handleUpload}
                disabled={isLoading || !word || isRecording}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-0 z-10"
              />
              <button
                disabled={isLoading || !word || isRecording}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-semibold bg-white dark:bg-secondary border border-border/80 shadow-sm hover:bg-muted text-foreground transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Upload size={18} />
                Upload
              </button>
            </div>
          </div>
        </div>
      </div>

      {!word && (
        <p className="text-xs text-orange-600/80 dark:text-orange-400/80 font-medium tracking-wide">
          Enter a target word to enable audio recording and upload.
        </p>
      )}

      {isRecording && (
        <div className="flex items-center gap-4 py-2 px-4 rounded-xl bg-red-50 border border-red-100 dark:bg-red-950/20 dark:border-red-900/30 animate-in fade-in slide-in-from-top-2">
          <div className="flex items-center gap-1.5 h-6">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div
                key={i}
                className="w-1.5 bg-red-500 rounded-full animate-pulse"
                style={{
                  height: `${Math.max(20, Math.random() * 100)}%`,
                  animationDuration: `${0.5 + Math.random() * 0.5}s`,
                  animationDelay: `${i * 0.1}s`,
                }}
              />
            ))}
          </div>
          <div className="flex items-center gap-2 font-mono text-sm font-semibold tracking-wider text-red-600 dark:text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
            {formatTime(recordingTime)} / 0:15
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex flex-col items-center justify-center py-8 mt-4 border border-border/40 rounded-xl bg-white/50 backdrop-blur-sm dark:bg-card/20 shadow-sm">
          <AudioLines className="w-10 h-10 text-orange-500 animate-pulse mb-4" />
          <p className="text-sm font-semibold tracking-wide text-foreground animate-pulse">
            Running Neural Phoneme Analysis...
          </p>
        </div>
      )}
    </div>
  );
}
