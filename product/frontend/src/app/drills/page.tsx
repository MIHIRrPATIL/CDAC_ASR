"use client";

import React, { useState, useEffect } from "react";
import { ProtectedRoute } from "../../components/protected-route";

import { GrainOverlay } from "../../components/grain-overlay";
import { getSRQueue, addSRCard, reviewSRCard, analyzeAudio, generateAIDrills, SpacedRepetitionItem, AIDrillResponse, getTTSAudioUrl } from "../../services/api";
import {
  Sparkles,
  BookOpen,
  Volume2,
  Mic,
  Activity,
  CheckCircle,
  HelpCircle,
  Clock,
  ArrowRight,
  Plus,
  Wand2,
  Loader2,
} from "lucide-react";
import AudioInput from "../../components/AudioInput";
import ResultsDashboard from "../../components/ResultsDashboard";

export default function DrillsPage() {
  const [activeTab, setActiveTab] = useState<"minimal" | "sr">("minimal");
  
  // Minimal Pairs State
  const [selectedPair, setSelectedPair] = useState<any | null>(null);
  const [selectedWord, setSelectedWord] = useState<string>("");
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  // AI Drill Generator State
  const [aiDrillPrompt, setAiDrillPrompt] = useState("");
  const [generatingDrills, setGeneratingDrills] = useState(false);
  const [aiGeneratedPairs, setAiGeneratedPairs] = useState<AIDrillResponse[]>([]);

  // Spaced Repetition State
  const [srQueue, setSrQueue] = useState<SpacedRepetitionItem[]>([]);
  const [currentCardIdx, setCurrentCardIdx] = useState<number>(0);
  const [newWord, setNewWord] = useState("");
  const [newPhonemes, setNewPhonemes] = useState("");
  const [srFeedback, setSrFeedback] = useState<any | null>(null);
  const [srAnalyzing, setSrAnalyzing] = useState(false);

  const minimalPairsList = [
    {
      label: "/v/ vs /w/",
      description: "Distinguish between labiodental fricative /v/ (teeth on lip) and bilabial approximant /w/ (rounded lips).",
      pairs: [
        { word1: "vest", word2: "west" },
        { word1: "vine", word2: "wine" },
        { word1: "vent", word2: "went" },
      ],
    },
    {
      label: "/s/ vs /z/",
      description: "Distinguish between voiceless /s/ and voiced alveolar fricative /z/ (vocal cords vibrating).",
      pairs: [
        { word1: "sip", word2: "zip" },
        { word1: "bus", word2: "buzz" },
        { word1: "price", word2: "prize" },
      ],
    },
    {
      label: "/ʈ/ vs /t̪/",
      description: "Practise retroflex /ʈ/ (tongue curled back) vs dental /t̪/ (tongue tip on front teeth).",
      pairs: [
        { word1: "tin", word2: "thin" },
        { word1: "team", word2: "theme" },
        { word1: "tick", word2: "thick" },
      ],
    },
  ];

  useEffect(() => {
    if (activeTab === "sr") {
      loadQueue();
    }
  }, [activeTab]);

  const loadQueue = async () => {
    try {
      const q = await getSRQueue();
      setSrQueue(q);
      setCurrentCardIdx(0);
    } catch (err) {
      console.error("Failed to load SR queue:", err);
    }
  };

  const handleMinimalPairSubmit = async (file: File, target: string) => {
    setAnalyzing(true);
    setAnalysisResult(null);
    try {
      const res = await analyzeAudio(file, target);
      setAnalysisResult(res);
    } catch (err) {
      console.error("Inference failed:", err);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSRSubmit = async (file: File, target: string) => {
    setSrAnalyzing(true);
    setSrFeedback(null);
    try {
      const res = await analyzeAudio(file, target);
      setSrFeedback(res);
      
      // Update Spaced Repetition SM-2 schedule on the backend
      const card = srQueue[currentCardIdx];
      await reviewSRCard(card.id, res.scores.phoneme);
    } catch (err) {
      console.error("SR review failed:", err);
    } finally {
      setSrAnalyzing(false);
    }
  };

  const playTTS = (text: string) => {
    const url = getTTSAudioUrl(text, false);
    const audio = new Audio(url);
    audio.play().catch(e => console.error("TTS play failed:", e));
  };

  const handleAddNewSR = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newWord) return;
    try {
      await addSRCard(newWord, newPhonemes);
      setNewWord("");
      setNewPhonemes("");
      loadQueue();
    } catch (err) {
      console.error("Failed to add SR card:", err);
    }
  };

  const currentCard = srQueue[currentCardIdx];

  const handleGenerateAIDrills = async () => {
    if (!aiDrillPrompt.trim()) return;
    setGeneratingDrills(true);
    try {
      const result = await generateAIDrills(aiDrillPrompt);
      setAiGeneratedPairs(prev => [result, ...prev]);
      setAiDrillPrompt("");
    } catch (err) {
      console.error("Failed to generate AI drills:", err);
    } finally {
      setGeneratingDrills(false);
    }
  };

  const allPairs = [
    ...aiGeneratedPairs.map(d => ({ label: d.label, description: d.description, pairs: d.pairs, isAI: true })),
    ...minimalPairsList.map(p => ({ ...p, isAI: false })),
  ];

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background text-foreground">

        <GrainOverlay />

        <main className="pt-32 pb-20 px-6 sm:px-12 max-w-6xl mx-auto">
          {/* Header */}
          <section className="mb-10 text-center">
            <h1 className="text-4xl font-bold tracking-tight text-foreground mb-4">
              Fluency Practice Drills
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto font-medium">
              Improve phonetic precision through dedicated structural minimal pair comparison exercises and spaced repetition intervals.
            </p>
          </section>

          {/* Tab Selection */}
          <div className="flex justify-center mb-10">
            <div className="flex bg-zinc-900/50 p-1.5 rounded-xl border border-border/40 backdrop-blur-md">
              <button
                onClick={() => {
                  setActiveTab("minimal");
                  setAnalysisResult(null);
                  setSelectedPair(null);
                }}
                className={`px-6 py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all cursor-pointer ${
                  activeTab === "minimal" ? "bg-orange-500 text-white shadow" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Minimal Pairs Drills
              </button>
              <button
                onClick={() => {
                  setActiveTab("sr");
                  setSrFeedback(null);
                }}
                className={`px-6 py-2.5 rounded-lg text-sm font-semibold tracking-wide transition-all cursor-pointer ${
                  activeTab === "sr" ? "bg-orange-500 text-white shadow" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Spaced Repetition Cards
              </button>
            </div>
          </div>

          {/* Tabs Content */}
          {activeTab === "minimal" ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Left Column: List of minimal pairs */}
              <div className="flex flex-col gap-4 lg:col-span-1">

                {/* AI Drill Generator */}
                <div className="p-5 rounded-2xl border border-orange-500/20 bg-orange-500/5 backdrop-blur-sm flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <Wand2 size={15} className="text-orange-500" />
                    <span className="font-bold text-sm text-foreground">AI Card Generator</span>
                  </div>
                  <p className="text-xs text-muted-foreground leading-relaxed">Type a phoneme contrast (e.g. <span className="font-mono text-orange-500">"r vs l"</span> or <span className="font-mono text-orange-500">"p vs b"</span>) and let AI create custom minimal pairs.</p>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={aiDrillPrompt}
                      onChange={e => setAiDrillPrompt(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && handleGenerateAIDrills()}
                      placeholder="e.g. th vs d, s vs sh..."
                      className="flex-1 px-3 py-2 bg-background border border-border/40 rounded-xl text-xs focus:outline-none focus:ring-2 focus:ring-orange-500/30 text-foreground"
                    />
                    <button
                      onClick={handleGenerateAIDrills}
                      disabled={generatingDrills || !aiDrillPrompt.trim()}
                      className="px-3.5 py-2 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white rounded-xl text-xs font-bold transition-all cursor-pointer shrink-0 flex items-center gap-1.5"
                    >
                      {generatingDrills ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
                      {generatingDrills ? "" : "Generate"}
                    </button>
                  </div>
                </div>

                <h3 className="font-mono text-xs uppercase tracking-widest text-muted-foreground mt-1">Select Confusion Pair</h3>
                {allPairs.map((pair, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setSelectedPair(pair);
                      setSelectedWord("");
                      setAnalysisResult(null);
                    }}
                    className={`p-5 rounded-2xl border text-left transition-all ${
                      selectedPair?.label === pair.label
                        ? "bg-orange-500/10 border-orange-500/30 shadow-sm"
                        : "bg-card/40 border-border/20 hover:border-border/40 hover:bg-card/60"
                    } cursor-pointer`}
                  >
                    <div className="flex items-center gap-2 mb-1.5">
                      {(pair as any).isAI && <span className="text-[9px] font-mono font-black text-orange-500 bg-orange-500/10 border border-orange-500/20 px-1.5 py-0.5 rounded">AI</span>}
                      <h4 className="font-bold text-base text-foreground">{pair.label}</h4>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">{pair.description}</p>
                  </button>
                ))}
              </div>

              {/* Middle/Right Columns: Exercise Panel */}
              <div className="lg:col-span-2 flex flex-col gap-6">
                {selectedPair ? (
                  <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex flex-col gap-6">
                    <div>
                      <h3 className="text-xl font-bold text-foreground mb-1">Practice Words</h3>
                      <p className="text-xs text-muted-foreground">Select a word and record your pronunciation to test accuracy differences.</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedPair.pairs.map((p: any, idx: number) => (
                        <div key={idx} className="flex gap-3 items-center justify-between p-4 rounded-xl border border-border/20 bg-secondary/10">
                          <button
                            onClick={() => {
                              setSelectedWord(p.word1);
                              setAnalysisResult(null);
                            }}
                            className={`flex-1 py-3 px-4 rounded-lg font-bold font-mono text-center border capitalize transition-all cursor-pointer ${
                              selectedWord === p.word1
                                ? "bg-orange-500 text-white border-orange-500"
                                : "bg-card/50 border-border/20 hover:bg-card text-foreground"
                            }`}
                          >
                            {p.word1}
                          </button>
                          <span className="text-xs font-mono text-muted-foreground font-semibold">vs</span>
                          <button
                            onClick={() => {
                              setSelectedWord(p.word2);
                              setAnalysisResult(null);
                            }}
                            className={`flex-1 py-3 px-4 rounded-lg font-bold font-mono text-center border capitalize transition-all cursor-pointer ${
                              selectedWord === p.word2
                                ? "bg-orange-500 text-white border-orange-500"
                                : "bg-card/50 border-border/20 hover:bg-card text-foreground"
                            }`}
                          >
                            {p.word2}
                          </button>
                        </div>
                      ))}
                    </div>

                    {selectedWord && (
                      <div className="mt-4 pt-6 border-t border-border/20 flex flex-col gap-6">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold text-muted-foreground">Target Word: <span className="text-foreground capitalize font-bold font-mono">"{selectedWord}"</span></span>
                          <button
                            onClick={() => playTTS(selectedWord)}
                            className="flex items-center gap-1.5 text-xs text-orange-500 font-semibold cursor-pointer hover:underline"
                          >
                            <Volume2 size={14} />
                            Listen Pronunciation
                          </button>
                        </div>

                        <AudioInput onAudioSubmit={handleMinimalPairSubmit} isLoading={analyzing} targetWord={selectedWord} />

                        {analysisResult && (
                          <div className="mt-4 border-t border-border/20 pt-6">
                            <ResultsDashboard data={analysisResult} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="p-12 bg-card/40 border border-border/20 rounded-3xl text-center text-muted-foreground flex flex-col items-center justify-center min-h-[300px]">
                    <HelpCircle className="w-12 h-12 text-zinc-600 mb-4 animate-bounce" />
                    <h3 className="font-bold text-foreground mb-2">No Confusion Pair Selected</h3>
                    <p className="text-xs max-w-xs leading-relaxed">Choose an acoustic confusion pair from the left panel to begin your target phoneme drill.</p>
                  </div>
                )}
              </div>
            </div>
          ) : (
            // Spaced Repetition Panel
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Queue List and Add panel */}
              <div className="lg:col-span-1 flex flex-col gap-6">
                {/* Add deck card */}
                <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm">
                  <h3 className="font-bold text-lg text-foreground mb-4 flex items-center gap-2">
                    <Plus size={18} className="text-orange-500" /> Add to Queue
                  </h3>
                  <form onSubmit={handleAddNewSR} className="flex flex-col gap-4">
                    <div>
                      <label className="block text-xs font-mono uppercase tracking-wider text-muted-foreground mb-1.5">Word</label>
                      <input
                        type="text"
                        value={newWord}
                        onChange={(e) => setNewWord(e.target.value)}
                        placeholder="e.g. specifically"
                        className="w-full px-3.5 py-2.5 bg-secondary/20 shadow-inner border border-border/40 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-orange-500/40 text-foreground"
                      />
                    </div>
                    <button
                      type="submit"
                      disabled={!newWord}
                      className="w-full py-2.5 rounded-xl font-bold bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white transition-all text-sm cursor-pointer"
                    >
                      Add Flashcard
                    </button>
                  </form>
                </div>

                {/* Queue status */}
                <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex flex-col gap-4">
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest flex items-center gap-2">
                    <Clock size={16} /> Deck Status
                  </h3>
                  <div className="flex items-center justify-between py-2 border-b border-border/10">
                    <span className="text-sm font-medium">Due for Review</span>
                    <span className="text-lg font-black text-orange-500">{srQueue.length}</span>
                  </div>
                </div>
              </div>

              {/* Study interface */}
              <div className="lg:col-span-2">
                {srQueue.length > 0 && currentCard ? (
                  <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex flex-col gap-6">
                    <div className="flex items-center justify-between border-b border-border/10 pb-4">
                      <span className="text-xs font-mono text-muted-foreground font-bold">CARD {currentCardIdx + 1} OF {srQueue.length}</span>
                      <span className="text-xs font-mono text-zinc-500 bg-secondary px-2.5 py-0.5 rounded-md">Ease: {currentCard.easeFactor}x</span>
                    </div>

                    <div className="text-center py-10 flex flex-col items-center gap-4">
                      <span className="text-4xl font-extrabold font-mono tracking-tight text-foreground capitalize">
                        {currentCard.word}
                      </span>
                      <button
                        onClick={() => playTTS(currentCard.word)}
                        className="flex items-center gap-2 text-sm text-orange-500 hover:text-orange-600 font-semibold cursor-pointer"
                      >
                        <Volume2 size={16} /> Play Guide
                      </button>
                    </div>

                    <div className="pt-6 border-t border-border/10 flex flex-col gap-6">
                      <AudioInput onAudioSubmit={handleSRSubmit} isLoading={srAnalyzing} targetWord={currentCard.word} />

                      {srFeedback && (
                        <div className="mt-4 border-t border-border/20 pt-6 flex flex-col gap-4">
                          <ResultsDashboard data={srFeedback} />
                          <button
                            onClick={() => {
                              setSrFeedback(null);
                              if (currentCardIdx + 1 < srQueue.length) {
                                setCurrentCardIdx(prev => prev + 1);
                              } else {
                                loadQueue();
                              }
                            }}
                            className="flex items-center gap-2 justify-center py-3 bg-indigo-500 hover:bg-indigo-600 text-white rounded-xl font-bold transition-all text-sm mt-4 cursor-pointer"
                          >
                            Next Card <ArrowRight size={16} />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="p-12 bg-card/40 border border-border/20 rounded-3xl text-center text-muted-foreground flex flex-col items-center justify-center min-h-[300px]">
                    <CheckCircle className="w-12 h-12 text-emerald-500 mb-4 animate-pulse" />
                    <h3 className="font-bold text-foreground mb-2">Deck Fully Reviewed!</h3>
                    <p className="text-xs max-w-xs leading-relaxed">No flashcards left in your spaced repetition queue. Add custom words on the left panel to build your vocabulary deck.</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </ProtectedRoute>
  );
}
