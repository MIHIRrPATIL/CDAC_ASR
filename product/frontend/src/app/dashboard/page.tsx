"use client";

import React, { useEffect, useState } from "react";

import { GrainOverlay } from "../../components/grain-overlay";
import { ProtectedRoute } from "../../components/protected-route";
import { useRouter } from "next/navigation";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  History,
  TrendingUp,
  BookOpen,
  Award,
  Clock,
  ArrowRight,
  Sparkles,
  Mic,
  Activity as ActivityIcon,
  Flame,
} from "lucide-react";
import { getDashboardStats, DashboardStats } from "../../services/api";

export default function DashboardPage() {
  const router = useRouter();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [isDemo, setIsDemo] = useState(false);

  // Fallback Mock Data for demo mode if user has 0 entries
  const demoHistory = [
    { date: "Mon", score: 65 },
    { date: "Tue", score: 68 },
    { date: "Wed", score: 74 },
    { date: "Thu", score: 72 },
    { date: "Fri", score: 85 },
    { date: "Sat", score: 88 },
    { date: "Sun", score: 92 },
  ];

  const demoHeatmap = [
    { phoneme: "ʈ", accuracy: 45, total_practiced: 12 },
    { phoneme: "ɖ", accuracy: 52, total_practiced: 8 },
    { phoneme: "ɲ", accuracy: 58, total_practiced: 15 },
    { phoneme: "ʉː", accuracy: 64, total_practiced: 10 },
    { phoneme: "ə", accuracy: 71, total_practiced: 22 },
    { phoneme: "ʃ", accuracy: 75, total_practiced: 18 },
  ];

  useEffect(() => {
    async function loadStats() {
      try {
        const data = await getDashboardStats();
        if (!data.history || data.history.length === 0) {
          setIsDemo(true);
          setStats({
            overall_accuracy: 78,
            practice_seconds: 15120, // 4.2 hours
            daily_streak: 5,
            global_rank: 8,
            history: demoHistory,
            heatmap: demoHeatmap,
          });
        } else {
          setIsDemo(false);
          setStats(data);
        }
      } catch (err) {
        console.error("Failed to load dashboard stats, enabling demo mode:", err);
        setIsDemo(true);
        setStats({
          overall_accuracy: 78,
          practice_seconds: 15120,
          daily_streak: 5,
          global_rank: 8,
          history: demoHistory,
          heatmap: demoHeatmap,
        });
      } finally {
        setLoading(false);
      }
    }
    loadStats();
  }, []);

  if (loading) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen bg-background flex items-center justify-center">

          <GrainOverlay />
          <div className="flex flex-col items-center gap-4 text-center">
            <div className="w-12 h-12 rounded-full border-4 border-orange-500/20 border-t-orange-500 animate-spin" />
            <p className="text-muted-foreground font-mono text-sm tracking-wider">COMPILING ANALYTICS...</p>
          </div>
        </div>
      </ProtectedRoute>
    );
  }

  const activeHistory = stats?.history || [];
  const activeHeatmap = stats?.heatmap || [];
  const practiceHours = stats ? (stats.practice_seconds / 3600).toFixed(1) : "0.0";

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background text-foreground">

        <GrainOverlay />

        <main className="pt-32 pb-20 px-6 sm:px-12 max-w-7xl mx-auto">
          {/* Header */}
          <section className="mb-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-foreground mb-2 flex items-center gap-3">
                  Your Progress Dashboard
                  {isDemo && (
                    <span className="text-xs font-mono uppercase bg-orange-500/10 text-orange-400 border border-orange-500/20 px-2 py-0.5 rounded-md">
                      Demo Mode
                    </span>
                  )}
                </h1>
                <p className="text-muted-foreground font-medium flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-orange-500" />
                  Track your phonetic improvement and custom target sounds
                </p>
              </div>
              <button 
                onClick={() => router.push("/analyzer")}
                className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold bg-orange-500 hover:bg-orange-600 text-white shadow-md transition-all hover:-translate-y-0.5 cursor-pointer"
              >
                <Mic size={18} />
                Start Practice session
              </button>
            </div>
          </section>

          {/* Bento Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Chart (Spans 2 columns) */}
            <div className="lg:col-span-2 p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-[14px] bg-emerald-500/10">
                    <ActivityIcon className="w-5 h-5 text-emerald-500" />
                  </div>
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Overall Accuracy Trend
                  </h3>
                </div>
                <span className="text-2xl font-black text-emerald-500">
                  {stats?.overall_accuracy}%
                </span>
              </div>
              
              <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={activeHistory}
                    margin={{ top: 0, right: 0, left: -20, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient
                        id="colorScore"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor="#10b981"
                          stopOpacity={0.3}
                        />
                        <stop
                          offset="95%"
                          stopColor="#10b981"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="hsl(var(--border))"
                      strokeOpacity={0.4}
                    />
                    <XAxis
                      dataKey="date"
                      axisLine={false}
                      tickLine={false}
                      tick={{
                        fontSize: 12,
                        fill: "hsl(var(--muted-foreground))",
                      }}
                    />
                    <YAxis
                      domain={[0, 100]}
                      axisLine={false}
                      tickLine={false}
                      tick={{
                        fontSize: 12,
                        fill: "hsl(var(--muted-foreground))",
                      }}
                    />
                    <Tooltip
                      contentStyle={{
                        borderRadius: "12px",
                        border: "1px solid hsl(var(--border))",
                        boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#10b981"
                      strokeWidth={3}
                      fillOpacity={1}
                      fill="url(#colorScore)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Quick Stats Column */}
            <div className="flex flex-col gap-6">
              {/* Daily Streak */}
              <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm bg-linear-to-br from-orange-500/10 to-transparent rounded-[24px] border border-border/40 shadow-sm flex-1 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-2">
                  <Flame className="w-5 h-5 text-orange-500" />
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Daily Streak
                  </h3>
                </div>
                <div className="text-[3rem] font-black text-foreground leading-none tracking-tighter">
                  {stats?.daily_streak} <span className="text-2xl text-muted-foreground font-normal">days</span>
                </div>
              </div>

              {/* Practice Time */}
              <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm bg-linear-to-br from-blue-500/10 to-transparent rounded-[24px] border border-border/40 shadow-sm flex-1 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-2">
                  <Clock className="w-5 h-5 text-blue-500" />
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Practice Time
                  </h3>
                </div>
                <div className="text-[3rem] font-black text-foreground leading-none tracking-tighter">
                  {practiceHours} <span className="text-2xl text-muted-foreground font-normal">hrs</span>
                </div>
              </div>
            </div>

            {/* Phoneme Weakness Heatmap Section (Spans 2 columns) */}
            <div className="lg:col-span-2 p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-[14px] bg-red-500/10">
                    <TrendingUp className="w-5 h-5 text-red-500" />
                  </div>
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Weak Phonemes Heatmap
                  </h3>
                </div>
                <span className="text-xs font-mono text-zinc-500">Sorted by lowest score</span>
              </div>

              {activeHeatmap.length === 0 ? (
                <div className="text-center py-8 text-zinc-500">
                  Record more pronunciation entries to compile heatmap data!
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                  {activeHeatmap.map((item, idx) => {
                    let colorClass = "bg-red-500/10 text-red-500 border-red-500/20";
                    if (item.accuracy >= 75) {
                      colorClass = "bg-emerald-500/10 text-emerald-500 border-emerald-500/20";
                    } else if (item.accuracy >= 55) {
                      colorClass = "bg-amber-500/10 text-amber-500 border-amber-500/20";
                    }

                    return (
                      <div
                        key={idx}
                        className={`p-4 rounded-xl border flex flex-col items-center justify-center text-center transition-all duration-300 hover:scale-[1.03] ${colorClass}`}
                      >
                        <span className="text-2xl font-bold font-mono">/{item.phoneme}/</span>
                        <span className="text-sm font-black mt-2">{item.accuracy}% Accuracy</span>
                        <span className="text-[10px] opacity-70 mt-0.5">Practiced: {item.total_practiced}x</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Quick Actions / Recommendations */}
            <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <div className="p-2 rounded-[14px] bg-purple-500/10">
                    <BookOpen className="w-5 h-5 text-purple-500" />
                  </div>
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Learning Actions
                  </h3>
                </div>

                <div className="flex flex-col gap-4">
                  <button 
                    onClick={() => router.push("/drills")}
                    className="flex items-center justify-between p-4 rounded-xl border border-border/40 bg-secondary/20 hover:bg-secondary/40 text-left transition-colors cursor-pointer group"
                  >
                    <div>
                      <h4 className="font-bold text-foreground">Minimal Pairs Drills</h4>
                      <p className="text-xs text-muted-foreground">Target specific phoneme swaps</p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
                  </button>

                  <button 
                    onClick={() => router.push("/drills?tab=sr")}
                    className="flex items-center justify-between p-4 rounded-xl border border-border/40 bg-secondary/20 hover:bg-secondary/40 text-left transition-colors cursor-pointer group"
                  >
                    <div>
                      <h4 className="font-bold text-foreground">Spaced Repetition</h4>
                      <p className="text-xs text-muted-foreground">Review weak words in intervals</p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
                  </button>

                  <button 
                    onClick={() => router.push("/ai-agent")}
                    className="flex items-center justify-between p-4 rounded-xl border border-border/40 bg-secondary/20 hover:bg-secondary/40 text-left transition-colors cursor-pointer group"
                  >
                    <div>
                      <h4 className="font-bold text-foreground">AI Agent Interview</h4>
                      <p className="text-xs text-muted-foreground">Mock dialogue voice partner</p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-foreground group-hover:translate-x-0.5 transition-all" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  );
}
