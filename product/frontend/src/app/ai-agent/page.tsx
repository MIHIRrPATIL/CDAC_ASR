"use client";

import React, { useState, useRef, useEffect } from "react";
import { ProtectedRoute } from "../../components/protected-route";
import { CustomCursor } from "../../components/custom-cursor";
import { GrainOverlay } from "../../components/grain-overlay";
import { postRoleplay, startRoleplay, analyzeAudio, getTTSAudioUrl, RoleplayResponse } from "../../services/api";
import {
  Volume2,
  Mic,
  Square,
  Sparkles,
  User,
  Bot,
  AlertCircle,
  RefreshCw,
  Activity,
  CheckCircle2,
  Coffee,
  Briefcase,
  ShoppingCart,
  Plane,
  MessageSquare,
  ChevronRight,
  Loader2,
  BookOpen,
  Send,
} from "lucide-react";
import ResultsDashboard from "../../components/ResultsDashboard";

interface ChatMessage {
  is_user: boolean;
  text: string;
  corrections?: string;
  analysis?: any;
  isRead?: boolean; // was this message read from a suggestion card?
}

// Scenario definitions
const SCENARIOS = [
  {
    id: "cafe",
    label: "Café Order",
    icon: Coffee,
    color: "orange",
    description: "Order drinks and food at a busy café counter.",
    prompt: "Two friends at a café counter, ordering drinks and chatting about their day",
  },
  {
    id: "interview",
    label: "Job Interview",
    icon: Briefcase,
    color: "indigo",
    description: "Practice professional speech in a mock job interview.",
    prompt: "A friendly job interview for a junior software position at a tech startup",
  },
  {
    id: "shopping",
    label: "Shopping Help",
    icon: ShoppingCart,
    color: "emerald",
    description: "Ask a store assistant for help finding items.",
    prompt: "Asking a helpful store assistant for product recommendations in a clothing store",
  },
  {
    id: "travel",
    label: "Airport Check-in",
    icon: Plane,
    color: "sky",
    description: "Navigate check-in, boarding passes and gate questions.",
    prompt: "A passenger checking in at an airport counter with flight queries",
  },
  {
    id: "friends",
    label: "Friends Chat",
    icon: MessageSquare,
    color: "pink",
    description: "Casual conversation between two old friends meeting up.",
    prompt: "Two close friends catching up after a long time, discussing life updates and funny stories",
  },
];

const COLOR_MAP: Record<string, { bg: string; border: string; text: string; badge: string; glow: string }> = {
  orange: { bg: "bg-orange-500/10", border: "border-orange-500/30", text: "text-orange-500", badge: "bg-orange-500", glow: "shadow-orange-500/20" },
  indigo: { bg: "bg-indigo-500/10", border: "border-indigo-500/30", text: "text-indigo-500", badge: "bg-indigo-500", glow: "shadow-indigo-500/20" },
  emerald: { bg: "bg-emerald-500/10", border: "border-emerald-500/30", text: "text-emerald-500", badge: "bg-emerald-500", glow: "shadow-emerald-500/20" },
  sky: { bg: "bg-sky-500/10", border: "border-sky-500/30", text: "text-sky-500", badge: "bg-sky-500", glow: "shadow-sky-500/20" },
  pink: { bg: "bg-pink-500/10", border: "border-pink-500/30", text: "text-pink-500", badge: "bg-pink-500", glow: "shadow-pink-500/20" },
};

export default function AIAgentPage() {
  const [screen, setScreen] = useState<"picker" | "chat">("picker");
  const [selectedScenario, setSelectedScenario] = useState<typeof SCENARIOS[0] | null>(null);
  const [customScenario, setCustomScenario] = useState("");

  const [dialogueHistory, setDialogueHistory] = useState<ChatMessage[]>([]);
  const [suggestedReplies, setSuggestedReplies] = useState<string[]>([]);
  const [selectedReply, setSelectedReply] = useState<string | null>(null);
  const [customUserReply, setCustomUserReply] = useState("");

  const [avatarState, setAvatarState] = useState<"IDLE" | "SPEAKING" | "LISTENING" | "THINKING">("IDLE");
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [loadingResponse, setLoadingResponse] = useState(false);
  const [loadingStart, setLoadingStart] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState<any | null>(null);
  const [lastCorrections, setLastCorrections] = useState<string>("");

  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [dialogueHistory, suggestedReplies]);

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

  const playTTS = (text: string) => {
    const url = getTTSAudioUrl(text, false);
    const audio = new Audio(url);
    setAvatarState("SPEAKING");
    audio.onended = () => setAvatarState("IDLE");
    audio.onerror = () => setAvatarState("IDLE");
    audio.play().catch(() => setAvatarState("IDLE"));
  };

  const startScenario = async () => {
    if (!selectedScenario && !customScenario.trim()) return;
    setLoadingStart(true);
    const scenarioText = selectedScenario?.prompt || customScenario.trim();

    try {
      const res = await startRoleplay(scenarioText);
      setDialogueHistory([{ is_user: false, text: res.response }]);
      setSuggestedReplies(res.suggested_replies || []);
      setScreen("chat");
      setTimeout(() => playTTS(res.response), 400);
    } catch (err) {
      console.error("Failed to start scenario:", err);
      // Fallback
      setDialogueHistory([{
        is_user: false,
        text: `Hello! Let's practice this scenario: "${scenarioText}". How would you like to start?`,
      }]);
      setSuggestedReplies([
        "Hi! I'm ready to begin, let's go.",
        "Hello, nice to meet you! Where do we start?",
        "Hey! I've been looking forward to this conversation.",
      ]);
      setScreen("chat");
    } finally {
      setLoadingStart(false);
    }
  };

  const startRecording = async () => {
    if (!selectedReply) return; // Must select a reply card first
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: "audio/wav" });
        const file = new File([audioBlob], "response.wav", { type: "audio/wav" });
        stream.getTracks().forEach((track) => track.stop());
        await processUserAudio(file);
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setAvatarState("LISTENING");
    } catch (err) {
      console.error("Microphone permission denied:", err);
      setAvatarState("IDLE");
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
    }
  };

  const processUserAudio = async (file: File) => {
    if (!selectedReply) return;
    setLoadingResponse(true);
    setAvatarState("THINKING");

    try {
      // Score the user's speech against the selected reply card text
      const ASRres = await analyzeAudio(file, selectedReply);
      setSelectedAnalysis(ASRres);

      const isSuggested = suggestedReplies.includes(selectedReply);
      const userTurn: ChatMessage = {
        is_user: true,
        text: selectedReply,
        analysis: ASRres,
        isRead: isSuggested,
      };

      const updatedHistory: ChatMessage[] = [...dialogueHistory, userTurn];
      setDialogueHistory(updatedHistory);
      setSelectedReply(null);
      setCustomUserReply("");

      // Fetch Ava's response
      const scenarioText = selectedScenario?.prompt || customScenario;
      const response = await postRoleplay(updatedHistory, scenarioText);

      const avaTurn: ChatMessage = {
        is_user: false,
        text: response.response,
        corrections: response.corrections,
      };

      setDialogueHistory([...updatedHistory, avaTurn]);
      setSuggestedReplies(response.suggested_replies || []);
      setLastCorrections(response.corrections || "");

      setTimeout(() => playTTS(response.response), 300);
    } catch (err) {
      console.error("Failed to process conversation turn:", err);
    } finally {
      setLoadingResponse(false);
    }
  };

  // Send text-only (no recording/ASR) to advance conversation
  const sendTextOnly = async () => {
    const replyText = customUserReply.trim() || selectedReply;
    if (!replyText) return;
    setLoadingResponse(true);
    setAvatarState("THINKING");

    try {
      const userTurn: ChatMessage = {
        is_user: true,
        text: replyText,
        isRead: false,
      };

      const updatedHistory: ChatMessage[] = [...dialogueHistory, userTurn];
      setDialogueHistory(updatedHistory);
      setSelectedReply(null);
      setCustomUserReply("");

      const scenarioText = selectedScenario?.prompt || customScenario;
      const response = await postRoleplay(updatedHistory, scenarioText);

      const avaTurn: ChatMessage = {
        is_user: false,
        text: response.response,
        corrections: response.corrections,
      };

      setDialogueHistory([...updatedHistory, avaTurn]);
      setSuggestedReplies(response.suggested_replies || []);
      setLastCorrections(response.corrections || "");

      setTimeout(() => playTTS(response.response), 300);
    } catch (err) {
      console.error("Failed to send text-only turn:", err);
    } finally {
      setLoadingResponse(false);
    }
  };

  // Scenario Picker Screen
  if (screen === "picker") {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-background text-foreground">
          <CustomCursor />
          <GrainOverlay />
          <main className="pt-32 pb-20 px-6 sm:px-12 max-w-5xl mx-auto">
            <section className="mb-10 text-center">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-orange-500/10 border border-orange-500/20 rounded-full text-xs font-bold text-orange-500 mb-4">
                <Sparkles size={12} /> Scenario Pronunciation Coach
              </div>
              <h1 className="text-4xl font-extrabold tracking-tight text-foreground mb-3">
                Choose Your Conversation
              </h1>
              <p className="text-sm text-muted-foreground max-w-lg mx-auto leading-relaxed">
                Select a real-world scenario. Ava will start the conversation — you choose a reply card and read it aloud. Your pronunciation gets scored accurately.
              </p>
            </section>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
              {SCENARIOS.map((s) => {
                const c = COLOR_MAP[s.color];
                const Icon = s.icon;
                const isSelected = selectedScenario?.id === s.id;
                return (
                  <button
                    key={s.id}
                    onClick={() => {
                      setSelectedScenario(s);
                      setCustomScenario("");
                    }}
                    className={`p-5 rounded-2xl border text-left transition-all duration-200 cursor-pointer group ${
                      isSelected
                        ? `${c.bg} ${c.border} shadow-lg shadow-${s.color}-500/10`
                        : "bg-card/50 border-border/20 hover:border-border/50 hover:bg-card/80"
                    }`}
                  >
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3 transition-all ${
                      isSelected ? c.bg + " " + c.border + " border" : "bg-secondary/50"
                    }`}>
                      <Icon size={18} className={isSelected ? c.text : "text-muted-foreground"} />
                    </div>
                    <div className="flex items-center justify-between mb-1">
                      <h3 className="font-bold text-base text-foreground">{s.label}</h3>
                      {isSelected && <CheckCircle2 size={15} className={c.text} />}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{s.description}</p>
                  </button>
                );
              })}
            </div>

            {/* Custom scenario input */}
            <div className="p-5 bg-card/40 border border-border/30 rounded-2xl mb-8">
              <div className="flex items-center gap-2 mb-2">
                <BookOpen size={14} className="text-muted-foreground" />
                <span className="text-sm font-semibold text-foreground">Or describe your own scenario</span>
              </div>
              <input
                type="text"
                value={customScenario}
                onChange={e => {
                  setCustomScenario(e.target.value);
                  if (e.target.value) setSelectedScenario(null);
                }}
                placeholder="e.g. A patient talking to their doctor about a headache..."
                className="w-full px-4 py-3 bg-background border border-border/40 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/30 text-foreground"
              />
            </div>

            <div className="flex justify-center">
              <button
                onClick={startScenario}
                disabled={(!selectedScenario && !customScenario.trim()) || loadingStart}
                className="flex items-center gap-2.5 px-8 py-3.5 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-2xl font-bold text-sm transition-all shadow-lg shadow-orange-500/20 cursor-pointer"
              >
                {loadingStart ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                {loadingStart ? "Starting conversation..." : "Start Conversation"}
                {!loadingStart && <ChevronRight size={16} />}
              </button>
            </div>
          </main>
        </div>
      </ProtectedRoute>
    );
  }

  // Chat Screen
  const scenarioLabel = selectedScenario?.label || "Custom Scenario";
  const scenarioColor = selectedScenario ? COLOR_MAP[selectedScenario.color] : COLOR_MAP["orange"];
  const ScenarioIcon = selectedScenario?.icon || MessageSquare;

  const getAvatarRing = () => {
    if (avatarState === "SPEAKING") return "border-emerald-500 shadow-emerald-500/30";
    if (avatarState === "LISTENING") return "border-red-500 shadow-red-500/30 animate-pulse";
    if (avatarState === "THINKING") return "border-indigo-500 shadow-indigo-500/30";
    return "border-border";
  };

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background text-foreground">
        <CustomCursor />
        <GrainOverlay />

        <style dangerouslySetInnerHTML={{ __html: `
          @keyframes voice-wave {
            0%, 100% { transform: scaleY(0.3); }
            50% { transform: scaleY(1); }
          }
          @keyframes spin-slow {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .wave-bar { transform-origin: bottom; }
          .spin-orb { animation: spin-slow 8s linear infinite; }
        `}} />

        <div className="pt-28 pb-8 px-4 sm:px-8 max-w-6xl mx-auto flex flex-col gap-6">

          {/* Top bar */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-8 h-8 rounded-xl flex items-center justify-center border ${scenarioColor.bg} ${scenarioColor.border}`}>
                <ScenarioIcon size={15} className={scenarioColor.text} />
              </div>
              <div>
                <h1 className="font-extrabold text-lg text-foreground leading-tight">{scenarioLabel}</h1>
                <p className="text-xs text-muted-foreground">Pronunciation Coach · Guided Reading Mode</p>
              </div>
            </div>
            <button
              onClick={() => {
                setScreen("picker");
                setDialogueHistory([]);
                setSuggestedReplies([]);
                setSelectedReply(null);
                setSelectedAnalysis(null);
                setLastCorrections("");
                setAvatarState("IDLE");
              }}
              className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground hover:text-foreground border border-border/30 hover:border-border px-3 py-1.5 rounded-lg transition-all cursor-pointer"
            >
              <RefreshCw size={12} /> Change Scenario
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

            {/* Left: Avatar + Status */}
            <div className="lg:col-span-1 flex flex-col gap-4">
              
              {/* Ava Card */}
              <div className="p-6 bg-card/60 backdrop-blur-sm border border-border/30 rounded-3xl flex flex-col items-center gap-4 text-center shadow-sm">
                {/* Avatar orb */}
                <div className={`relative w-28 h-28 rounded-full border-4 shadow-xl transition-all duration-500 flex items-center justify-center ${getAvatarRing()}`}>
                  {avatarState === "THINKING" && (
                    <div className="absolute inset-0 rounded-full border-2 border-dashed border-indigo-500/40 spin-orb" />
                  )}
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-orange-400 to-amber-500 flex items-center justify-center shadow-inner">
                    <Bot size={28} className="text-white" />
                  </div>
                </div>

                <div>
                  <p className="font-extrabold text-base text-foreground">Ava</p>
                  <p className={`text-xs font-semibold mt-0.5 ${
                    avatarState === "SPEAKING" ? "text-emerald-500" :
                    avatarState === "LISTENING" ? "text-red-500" :
                    avatarState === "THINKING" ? "text-indigo-500" :
                    "text-muted-foreground"
                  }`}>
                    {avatarState === "SPEAKING" ? "▶ Speaking..." :
                     avatarState === "LISTENING" ? "● Recording..." :
                     avatarState === "THINKING" ? "◌ Analyzing..." :
                     "Waiting for you"}
                  </p>
                </div>

                {/* Waveform bars */}
                <div className="flex gap-1 items-end h-8 px-2">
                  {[3,5,2,7,4,6,3,8,5,4].map((h, i) => (
                    <div
                      key={i}
                      className={`w-1 rounded-full wave-bar transition-colors duration-300 ${
                        avatarState === "SPEAKING" ? "bg-emerald-500" :
                        avatarState === "LISTENING" ? "bg-red-500" :
                        "bg-border"
                      }`}
                      style={{
                        height: `${h * 10}%`,
                        animation: (avatarState === "SPEAKING" || avatarState === "LISTENING")
                          ? `voice-wave ${0.5 + i * 0.12}s ease-in-out infinite`
                          : "none",
                      }}
                    />
                  ))}
                </div>

                <div className="w-full pt-3 border-t border-border/20 text-left space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Turns</span>
                    <span className="font-bold text-foreground">{dialogueHistory.length}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Mode</span>
                    <span className="font-bold text-foreground">Guided Read</span>
                  </div>
                </div>
              </div>

              {/* Grammar corrections */}
              <div className="p-5 bg-card/60 backdrop-blur-sm border border-border/30 rounded-2xl flex flex-col gap-3 shadow-sm">
                <div className="flex items-center gap-2">
                  <AlertCircle size={14} className="text-amber-500" />
                  <span className="font-bold text-sm text-foreground">Ava's Feedback</span>
                </div>
                {lastCorrections ? (
                  <div className="p-3 rounded-xl border-l-4 border-amber-500 bg-amber-500/5 text-xs text-foreground leading-relaxed font-medium">
                    {lastCorrections}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    After you speak, Ava will give grammar and vocabulary feedback here.
                  </p>
                )}
              </div>
            </div>

            {/* Right: Chat + Reply Cards */}
            <div className="lg:col-span-2 flex flex-col gap-4">

              {/* Chat window */}
              <div className="bg-card/50 backdrop-blur-sm border border-border/30 rounded-3xl overflow-hidden shadow-sm flex flex-col" style={{ minHeight: "340px", maxHeight: "420px" }}>
                <div className="px-5 py-3 border-b border-border/20 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-orange-500" />
                  <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground font-bold">Conversation</span>
                </div>
                <div className="flex-1 p-5 overflow-y-auto space-y-4">
                  {dialogueHistory.map((msg, i) => (
                    <div
                      key={i}
                      className={`flex gap-2.5 max-w-[82%] ${msg.is_user ? "ml-auto flex-row-reverse" : "mr-auto"}`}
                    >
                      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 border ${
                        msg.is_user
                          ? "bg-gradient-to-tr from-orange-500 to-amber-500 text-white border-orange-400"
                          : "bg-card border-border/40 text-muted-foreground"
                      }`}>
                        {msg.is_user ? <User size={14} /> : <Bot size={14} />}
                      </div>
                      <div className="flex flex-col gap-1 max-w-full">
                        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed border shadow-sm ${
                          msg.is_user
                            ? "bg-orange-500 text-white border-orange-400 rounded-tr-sm"
                            : "bg-background border-border/30 text-foreground rounded-tl-sm"
                        }`}>
                          {msg.text}
                          {msg.isRead && (
                            <div className="mt-2 pt-2 border-t border-white/20 flex items-center gap-1 text-[10px] font-bold text-white/70">
                              <BookOpen size={9} /> Read aloud from suggestion card
                            </div>
                          )}
                        </div>
                        {!msg.is_user && (
                          <button
                            onClick={() => playTTS(msg.text)}
                            className="flex items-center gap-1 text-[10px] font-semibold text-muted-foreground hover:text-foreground w-max pl-1 cursor-pointer"
                          >
                            <Volume2 size={10} /> Play
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                  {loadingResponse && (
                    <div className="flex gap-2.5 items-center mr-auto">
                      <div className="w-8 h-8 rounded-full bg-card border border-border/40 flex items-center justify-center">
                        <Bot size={14} className="text-muted-foreground" />
                      </div>
                      <div className="flex gap-1.5 px-4 py-3 bg-background border border-border/30 rounded-2xl rounded-tl-sm">
                        <span className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce" />
                        <span className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:0.15s]" />
                        <span className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full animate-bounce [animation-delay:0.3s]" />
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
              </div>

              {/* Suggested Reply Cards */}
              {suggestedReplies.length > 0 && !loadingResponse && (
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono uppercase tracking-widest text-muted-foreground font-bold">Your Turn — Select a reply to read aloud</span>
                  </div>
                  <div className="flex flex-col gap-2.5">
                    {suggestedReplies.map((reply, i) => {
                      const isSelected = selectedReply === reply;
                      return (
                        <button
                          key={i}
                          onClick={() => {
                            setSelectedReply(isSelected ? null : reply);
                            setCustomUserReply("");
                          }}
                          className={`w-full text-left p-4 rounded-2xl border transition-all duration-200 cursor-pointer group ${
                            isSelected
                              ? "bg-orange-500/10 border-orange-500/40 shadow-md shadow-orange-500/10"
                              : "bg-card/50 border-border/20 hover:border-border/50 hover:bg-card/80"
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            <div className={`mt-0.5 w-6 h-6 rounded-lg flex items-center justify-center text-xs font-black shrink-0 transition-all ${
                              isSelected ? "bg-orange-500 text-white" : "bg-secondary text-muted-foreground"
                            }`}>
                              {i + 1}
                            </div>
                            <p className={`text-sm leading-relaxed font-medium flex-1 ${isSelected ? "text-foreground" : "text-muted-foreground group-hover:text-foreground"}`}>
                              "{reply}"
                            </p>
                            {isSelected && <CheckCircle2 size={16} className="text-orange-500 shrink-0 mt-0.5" />}
                          </div>
                          {isSelected && (
                            <p className="text-[10px] text-orange-500/70 font-semibold mt-2 ml-9">
                              ✓ Selected · Press record below to speak this line
                            </p>
                          )}
                        </button>
                      );
                    })}

                    <div className="relative mt-2 flex gap-2">
                      <div className="relative flex-1">
                        <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
                          <Sparkles size={14} className="text-orange-500" />
                        </div>
                        <input
                          type="text"
                          value={customUserReply}
                          onChange={(e) => {
                            setCustomUserReply(e.target.value);
                            setSelectedReply(e.target.value || null);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && customUserReply.trim()) {
                              sendTextOnly();
                            }
                          }}
                          placeholder="Type your own response here (Enter to send without recording)..."
                          className="w-full pl-9 pr-4 py-3.5 bg-card/60 border border-border/25 rounded-2xl text-xs focus:outline-none focus:ring-2 focus:ring-orange-500/30 text-foreground placeholder:text-muted-foreground font-medium transition-all"
                        />
                      </div>
                      {customUserReply.trim() && (
                        <button
                          onClick={sendTextOnly}
                          disabled={loadingResponse}
                          className="px-4 py-3.5 bg-orange-500 hover:bg-orange-600 disabled:opacity-40 text-white rounded-2xl text-xs font-bold transition-all flex items-center gap-1.5 cursor-pointer shrink-0"
                        >
                          <Send size={12} /> Send
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Recording Controls */}
              <div className={`p-4 rounded-2xl border flex items-center justify-between gap-4 transition-all ${
                selectedReply
                  ? "bg-orange-500/5 border-orange-500/30"
                  : "bg-card/30 border-border/20 opacity-60"
              }`}>
                <div className="flex-1">
                  {!selectedReply ? (
                    <p className="text-xs text-muted-foreground font-medium">Select a reply card above, then press the mic to record — or type a custom reply and hit Enter.</p>
                  ) : isRecording ? (
                    <div className="flex items-center gap-2 text-red-500 text-xs font-mono font-bold animate-pulse">
                      <span className="w-2 h-2 rounded-full bg-red-500" />
                      RECORDING · {recordingTime}s — speak: "{selectedReply.slice(0, 40)}{selectedReply.length > 40 ? "..." : ""}"
                    </div>
                  ) : (
                    <p className="text-xs text-foreground font-semibold">
                      Ready to record: <span className="text-orange-500">"{selectedReply.slice(0, 50)}{selectedReply.length > 50 ? "..." : ""}"</span>
                    </p>
                  )}
                </div>
                <div>
                  {isRecording ? (
                    <button
                      onClick={stopRecording}
                      className="p-3.5 bg-red-500 hover:bg-red-600 text-white rounded-full shadow-lg transition-all hover:scale-105 cursor-pointer"
                    >
                      <Square size={16} />
                    </button>
                  ) : (
                    <button
                      onClick={startRecording}
                      disabled={!selectedReply || loadingResponse}
                      className="p-3.5 bg-orange-500 hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-full shadow-lg transition-all hover:scale-105 cursor-pointer"
                    >
                      <Mic size={16} />
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Acoustic Analysis Panel */}
          {selectedAnalysis && (
            <div className="border-t border-border/20 pt-6 flex flex-col gap-4">
              <div className="flex items-center gap-2">
                <Activity size={16} className="text-orange-500" />
                <h3 className="font-bold text-foreground">Acoustic Feedback — Last Reading</h3>
              </div>
              <ResultsDashboard data={selectedAnalysis} />
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}
