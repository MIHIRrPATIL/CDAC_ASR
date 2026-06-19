"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { CustomCursor } from "@/components/custom-cursor";
import { GrainOverlay } from "@/components/grain-overlay";
import { MagneticButton } from "@/components/magnetic-button";
import EtherBackground from "@/components/EtherBackground";
import { Activity, Mic, BarChart3, Target, Zap, BookOpen, Brain, Star } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    if (isAuthenticated) {
      router.replace("/dashboard");
    }
  }, [isAuthenticated, router]);

  return (
    <main className="relative min-h-screen w-full overflow-y-auto bg-black text-white selection:bg-orange-500/30 font-sans">
      <CustomCursor />
      <GrainOverlay />
      <EtherBackground />

      {/* Hero Section */}
      <section className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-20 text-center max-w-5xl mx-auto">
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-6 inline-flex items-center gap-2 rounded-full border border-orange-500/30 bg-orange-500/10 px-4 py-1.5 backdrop-blur-md"
        >
          <span className="flex h-2 w-2 rounded-full bg-orange-500 animate-pulse" />
          <p className="font-mono text-xs text-orange-400 uppercase tracking-widest font-semibold">
            AI-Powered Pronunciation Scoring
          </p>
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
          className="mb-8 font-light leading-[1.1] tracking-tight text-white text-5xl md:text-7xl lg:text-8xl"
        >
          Master your <br />
          <span className="text-orange-500 font-normal">spoken fluency.</span>
        </motion.h1>

        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mb-12 max-w-2xl text-lg leading-relaxed text-zinc-400 md:text-xl"
        >
          Analyze your pronunciation with clinical precision. Get real-time feedback 
          on phonemes, pitch, and duration using our advanced neural pipeline.
        </motion.p>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-col gap-4 sm:flex-row sm:items-center justify-center"
        >
          <div onClick={() => router.push("/analyzer")}>
            <MagneticButton size="lg" variant="primary">
              Start Analyzing
            </MagneticButton>
          </div>
          <div onClick={() => router.push("/auth")}>
            <MagneticButton size="lg" variant="secondary">
              Sign In
            </MagneticButton>
          </div>
        </motion.div>
      </section>

      {/* Choose Your Lab (New Layout workflows) */}
      <section className="relative z-10 py-24 px-6 max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">Choose Your Practice Lab</h2>
          <p className="font-mono text-orange-500/60 tracking-wider uppercase text-sm">/ FLEXIBLE PRONUNCIATION TRAINING</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Card 1: Single Word Lab */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="group flex flex-col justify-between p-8 rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-sm hover:border-orange-500/30 transition-all duration-500 shadow-xl"
          >
            <div>
              <div className="w-14 h-14 rounded-2xl bg-orange-500/20 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-300">
                <Mic className="text-orange-500 w-7 h-7" />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Single Word Analyzer</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6">
                Target tricky words, compound nouns, or specific technical terms. Get micro-level phoneme accuracy, IPA guides on hover, and acoustic scores for precise phonetics.
              </p>
            </div>
            <div className="mt-auto" onClick={() => router.push("/analyzer")}>
              <MagneticButton size="default" variant="secondary">
                Enter Word Lab
              </MagneticButton>
            </div>
          </motion.div>

          {/* Card 2: Sentence & Paragraph Reader */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="group flex flex-col justify-between p-8 rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-sm hover:border-orange-500/30 transition-all duration-500 shadow-xl"
          >
            <div>
              <div className="w-14 h-14 rounded-2xl bg-orange-500/20 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-300">
                <BookOpen className="text-orange-500 w-7 h-7" />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Paragraph Reader</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6">
                Practice reading full sentences, short stories, or diagnostic diagnostic paragraphs. Analyze sentence stress, speech rhythm, pitch contours, and receive custom overall feedback.
              </p>
            </div>
            <div className="mt-auto" onClick={() => router.push("/paragraphs")}>
              <MagneticButton size="default" variant="primary">
                Enter Paragraph Lab
              </MagneticButton>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Precision Metrics Section */}
      <section className="relative z-10 py-24 bg-zinc-950/60 border-y border-white/5">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl font-light text-white mb-4">Precision Feedback System</h2>
            <p className="font-mono text-orange-500/60 tracking-wider uppercase text-sm">/ STATE OF THE ART ACOUSTIC LOGIC</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { icon: Target, title: "Phonetic Accuracy", desc: "Forced alignment and Goodness of Pronunciation (GoP) mapping." },
              { icon: Zap, title: "Pitch Analysis", desc: "Dynamic Time Warping (DTW) to analyze intonation curves." },
              { icon: Activity, title: "Duration & Rhythm", desc: "Syllable duration ratios and timing correctness indicators." },
              { icon: Brain, title: "Actionable Feedback", desc: "Rule-based analysis engine generating concrete practice tips." }
            ].map((f, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="p-6 rounded-2xl border border-white/5 bg-zinc-900/20 backdrop-blur-xs hover:border-orange-500/20 hover:bg-zinc-900/30 transition-all duration-300"
              >
                <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center mb-6">
                  <f.icon className="text-orange-500" size={20} />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{f.title}</h3>
                <p className="text-xs text-zinc-500 leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="relative z-10 py-28 px-6 text-center max-w-4xl mx-auto">
        <motion.h2 
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-4xl md:text-5xl lg:text-6xl font-light text-white mb-6"
        >
          Ready to transform your speech?
        </motion.h2>
        <p className="text-lg text-zinc-400 mb-10 max-w-xl mx-auto">
          Sign up today and get immediate phoneme-level grading on words, sentences, and paragraphs.
        </p>
        <div className="flex justify-center" onClick={() => router.push("/auth")}>
          <MagneticButton size="lg" variant="primary">
            Create Free Account
          </MagneticButton>
        </div>
      </section>
    </main>
  );
}
