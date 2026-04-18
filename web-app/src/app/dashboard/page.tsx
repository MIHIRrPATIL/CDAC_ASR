"use client";

import React, { useState } from "react";
import { CustomCursor } from "../../components/custom-cursor";
import { GrainOverlay } from "../../components/grain-overlay";
import { ProtectedRoute } from "../../components/protected-route";
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
  MapPin,
  Clock,
  ArrowRight,
  Sparkles,
} from "lucide-react";

export default function DashboardPage() {
  const [mounted] = useState(true);

  // Mock Data: Recent pronunciation scores
  const historyData = [
    { date: "Mon", score: 65 },
    { date: "Tue", score: 68 },
    { date: "Wed", score: 74 },
    { date: "Thu", score: 72 },
    { date: "Fri", score: 85 },
    { date: "Sat", score: 88 },
    { date: "Sun", score: 92 },
  ];

  const recentRecordings = [
    { word: "because", date: "Today", score: 92, improvement: "+4%" },
    { word: "temperature", date: "Yesterday", score: 85, improvement: "+12%" },
    { word: "specifically", date: "Wednesday", score: 74, improvement: "+2%" },
  ];

  const learningPaths = [
    {
      title: "Mastering Fricatives",
      desc: "Focus on 's', 'z', 'f', 'v'",
      progress: 75,
      color: "emerald",
    },
    {
      title: "Rhythm & Stress",
      desc: "Word-level syllable intonation",
      progress: 40,
      color: "blue",
    },
    {
      title: "Vowel Transitions",
      desc: "Diphthong articulation",
      progress: 15,
      color: "purple",
    },
  ];

  if (!mounted) return null;

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        <CustomCursor />
        <GrainOverlay />

        <main className="pt-32 pb-20 px-6 sm:px-12 max-w-7xl mx-auto">
          {/* Header */}
          <section className="mb-10 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
              <div>
                <h1 className="text-4xl font-bold tracking-tight text-foreground mb-2">
                  Your Progress Dashboard
                </h1>
                <p className="text-muted-foreground font-medium flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-orange-500" />
                  Track your phonetic improvement over time
                </p>
              </div>
              <button className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold bg-orange-500 hover:bg-orange-600 text-white shadow-md transition-all hover:-translate-y-0.5">
                <TrendingUp size={18} />
                Generate Study Plan
              </button>
            </div>
          </section>

          {/* Bento Grid Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Main Chart (Spans 2 columns) */}
            <div className="lg:col-span-2 p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-[14px] bg-emerald-500/10">
                    <Activity className="w-5 h-5 text-emerald-500" />
                  </div>
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Overall Accuracy Trend
                  </h3>
                </div>
                <span className="text-2xl font-black text-emerald-500">
                  92%
                </span>
              </div>
              <div className="h-[280px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={historyData}
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
                      domain={["auto", "auto"]}
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
              <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm bg-linear-to-br from-blue-500/10 to-transparent rounded-[24px] border border-border/40 shadow-sm flex-1 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-2">
                  <Award className="w-5 h-5 text-blue-500" />
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Global Rank
                  </h3>
                </div>
                <div className="text-[3rem] font-black text-foreground leading-none tracking-tighter">
                  Top 8<span className="text-2xl text-muted-foreground">%</span>
                </div>
              </div>

              <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm bg-linear-to-br from-orange-500/10 to-transparent rounded-[24px] border border-border/40 shadow-sm flex-1 flex flex-col justify-center">
                <div className="flex items-center gap-3 mb-2">
                  <Clock className="w-5 h-5 text-orange-500" />
                  <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                    Practice Time
                  </h3>
                </div>
                <div className="text-[3rem] font-black text-foreground leading-none tracking-tighter">
                  4.2<span className="text-2xl text-muted-foreground">hrs</span>
                </div>
              </div>
            </div>

            {/* Learning Paths */}
            <div className="lg:col-span-2 p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-[14px] bg-purple-500/10">
                  <BookOpen className="w-5 h-5 text-purple-500" />
                </div>
                <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                  Recommended Learning Paths
                </h3>
              </div>

              <div className="flex flex-col gap-4">
                {learningPaths.map((path, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-4 rounded-xl border border-border/40 bg-secondary/20 hover:bg-secondary/40 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div
                        className={`w-12 h-12 rounded-full flex items-center justify-center bg-${path.color}-500/10 text-${path.color}-500`}
                      >
                        <MapPin size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-foreground">
                          {path.title}
                        </h4>
                        <p className="text-sm text-muted-foreground">
                          {path.desc}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 w-1/3">
                      <div className="flex-1 h-2 bg-muted/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full bg-${path.color}-500 rounded-full`}
                          style={{ width: `${path.progress}%` }}
                        />
                      </div>
                      <span className="font-bold text-sm">
                        {path.progress}%
                      </span>
                      <button className="p-2 rounded-lg hover:bg-muted text-muted-foreground hover:text-foreground transition-colors">
                        <ArrowRight size={18} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Recent History */}
            <div className="p-6 bg-white dark:bg-card/60 backdrop-blur-sm rounded-[24px] border border-border/40 shadow-sm flex-1">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 rounded-[14px] bg-foreground/5">
                  <History className="w-5 h-5 text-foreground" />
                </div>
                <h3 className="font-semibold text-muted-foreground text-sm uppercase tracking-widest">
                  Recent Recordings
                </h3>
              </div>

              <div className="flex flex-col gap-5">
                {recentRecordings.map((rec, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div>
                      <h4 className="font-bold font-mono text-foreground capitalize">
                        {rec.word}
                      </h4>
                      <p className="text-xs text-muted-foreground">
                        {rec.date}
                      </p>
                    </div>
                    <div className="text-right">
                      <div className="font-black text-lg text-foreground">
                        {rec.score}%
                      </div>
                      <div className="text-xs font-semibold text-emerald-500">
                        {rec.improvement}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </main>
      </div>
    </ProtectedRoute>
  );
}

function Activity(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" />
    </svg>
  );
}
