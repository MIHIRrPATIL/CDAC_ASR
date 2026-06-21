"use client";

import React, { useRef, useState, useEffect } from "react";
import {
  Mic,
  Upload,
  Square,
  Sparkles,
  AudioLines,
} from "lucide-react";
import { generateAIText } from "../services/api";

interface ParagraphInputProps {
  onAudioSubmit: (file: File, targetText: string) => void;
  isLoading: boolean;
}

const PRESETS = [
  {
    label: "Pangram",
    text: "The quick brown fox jumps over the lazy dog.",
  },
  {
    label: "Stella Diagnostic",
    text: "Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
  },
  {
    label: "Technical Prompt",
    text: "Today we will discuss the basic principles of neural networks and machine learning systems.",
  },
];

export default function ParagraphInput({
  onAudioSubmit,
  isLoading,
}: ParagraphInputProps) {
  const [text, setText] = useState("");
  const [topic, setTopic] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setIsGenerating(true);
    try {
      const res = await generateAIText(topic.trim());
      setText(res.paragraph);
    } catch (err) {
      console.error("Failed to generate AI paragraph:", err);
    } finally {
      setIsGenerating(false);
    }
  };

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
    if (file && text.trim()) {
      onAudioSubmit(file, text.trim());
    }
  };

  const toggleRecording = async () => {
    if (!text.trim()) return;

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
          onAudioSubmit(file, text.trim());
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
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-orange-500/10 dark:bg-orange-950/30">
            <Sparkles className="w-5 h-5 text-orange-600 dark:text-orange-500" />
          </div>
          <h2 className="text-xl font-bold text-foreground">Paragraph Acoustic Input</h2>
        </div>
      </div>

      {/* Preset prompt buttons */}
      <div className="flex flex-col gap-2">
        <label className="block text-xs font-semibold tracking-wider text-muted-foreground uppercase">
          Quick Preset Prompts
        </label>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              onClick={() => setText(preset.text)}
              disabled={isLoading || isRecording || isGenerating}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                text === preset.text
                  ? "bg-orange-500/10 border-orange-500/40 text-orange-600 dark:text-orange-400 font-semibold"
                  : "bg-white dark:bg-secondary/15 hover:bg-muted border-border/80 text-muted-foreground hover:text-foreground"
              }`}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* AI Generator Block */}
      <div className="p-5 rounded-2xl border border-orange-500/20 bg-orange-500/5 dark:bg-orange-950/20 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-orange-500 animate-pulse" />
          <span className="text-sm font-bold text-foreground">AI Paragraph Generator</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Enter a topic to generate a reading passage that incorporates your personal pronunciation weaknesses.
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            disabled={isGenerating || isLoading || isRecording}
            placeholder="e.g., job interview, space flight, requesting a coffee..."
            className="flex-1 px-3.5 py-2.5 bg-white dark:bg-zinc-900 border border-border/80 dark:border-border/40 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/50 text-foreground transition-all"
          />
          <button
            onClick={handleGenerate}
            disabled={!topic.trim() || isGenerating || isLoading || isRecording}
            className="px-5 py-2.5 rounded-xl text-xs font-bold bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white flex items-center gap-1.5 transition-all shadow-md shadow-orange-500/10 dark:shadow-none cursor-pointer"
          >
            {isGenerating ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles size={14} />
                Generate
              </>
            )}
          </button>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-6 items-start">
        {/* Text area for paragraphs */}
        <div className="w-full md:w-2/3">
          <label className="block text-xs font-semibold tracking-wider text-muted-foreground uppercase mb-2">
            Target Paragraph / Sentences
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={isLoading || isRecording}
            placeholder="Type or copy your paragraph here, or click one of the presets above..."
            rows={4}
            className="w-full px-4 py-3 bg-white dark:bg-secondary/20 shadow-sm border border-border/60 dark:border-border/40 rounded-xl focus:outline-none focus:ring-2 focus:ring-orange-500/50 text-foreground transition-all resize-none text-sm leading-relaxed"
          />
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-3 w-full md:w-1/3 md:self-end">
          <button
            onClick={toggleRecording}
            disabled={isLoading || !text.trim()}
            className={`w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold transition-all ${
              isRecording
                ? "bg-red-50 text-red-600 ring-1 ring-red-200 shadow-sm hover:bg-red-100 dark:bg-red-950/40 dark:ring-red-900/50 dark:text-red-400"
                : "bg-orange-500 hover:bg-orange-600 text-white shadow-md shadow-orange-500/10 dark:shadow-none hover:-translate-y-0.5"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {isRecording ? <Square size={18} /> : <Mic size={18} />}
            {isRecording ? "Stop Recording" : "Record Voice"}
          </button>

          <div className="relative w-full">
            <input
              type="file"
              accept="audio/wav,audio/*"
              onChange={handleUpload}
              disabled={isLoading || !text.trim() || isRecording}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed disabled:opacity-0 z-10"
            />
            <button
              disabled={isLoading || !text.trim() || isRecording}
              className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl font-semibold bg-white dark:bg-secondary border border-border/80 shadow-sm hover:bg-muted text-foreground transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Upload size={18} />
              Upload Audio File
            </button>
          </div>
        </div>
      </div>

      {!text.trim() && (
        <p className="text-xs text-orange-600/80 dark:text-orange-400/80 font-medium tracking-wide">
          Select a preset or type a paragraph to enable audio recording.
        </p>
      )}

      {isRecording && (
        <div className="flex items-center gap-4 py-2.5 px-4 rounded-xl bg-red-50 border border-red-100 dark:bg-red-950/20 dark:border-red-900/30 animate-in fade-in slide-in-from-top-2">
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
            {formatTime(recordingTime)} / 1:00
          </div>
        </div>
      )}

      {isLoading && (
        <div className="flex flex-col items-center justify-center py-10 mt-4 border border-border/40 rounded-xl bg-white/50 backdrop-blur-sm dark:bg-card/20 shadow-sm">
          <AudioLines className="w-10 h-10 text-orange-500 animate-pulse mb-4" />
          <p className="text-sm font-semibold tracking-wide text-foreground animate-pulse">
            Processing paragraph forced alignment...
          </p>
        </div>
      )}
    </div>
  );
}
