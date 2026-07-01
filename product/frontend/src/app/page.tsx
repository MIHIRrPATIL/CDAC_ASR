"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

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

      {/* Choose Your Lab (Features) */}
      <section id="features" className="relative z-10 py-28 px-6 max-w-6xl mx-auto scroll-mt-24">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">Choose Your Practice Lab</h2>
          <p className="font-mono text-orange-500/80 tracking-wider uppercase text-sm">/ FLEXIBLE PRONUNCIATION TRAINING</p>
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
                Target tricky words, compound nouns, or specific medical/technical terminology. Get micro-level phoneme accuracy, detailed IPA guides on hover, and acoustic scores for precise phonetics.
              </p>
              <ul className="text-xs text-zinc-500 space-y-2 mb-8">
                <li className="flex items-center gap-2">✔ IPA token comparison grid</li>
                <li className="flex items-center gap-2">✔ Target vs. spoken phonetic mismatch highlights</li>
                <li className="flex items-center gap-2">✔ Vowel/Consonant articulation feedback</li>
              </ul>
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
              <ul className="text-xs text-zinc-500 space-y-2 mb-8">
                <li className="flex items-center gap-2">✔ Prosody & pitch trajectory tracking</li>
                <li className="flex items-center gap-2">✔ Color-coded paragraph flow visualizer</li>
                <li className="flex items-center gap-2">✔ Syllable duration ratio analysis</li>
              </ul>
            </div>
            <div className="mt-auto" onClick={() => router.push("/paragraphs")}>
              <MagneticButton size="default" variant="primary">
                Enter Paragraph Lab
              </MagneticButton>
            </div>
          </motion.div>

          {/* Card 3: AI Tutor Cockpit */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="group flex flex-col justify-between p-8 rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-sm hover:border-orange-500/30 transition-all duration-500 shadow-xl"
          >
            <div>
              <div className="w-14 h-14 rounded-2xl bg-orange-500/20 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-300">
                <Brain className="text-orange-500 w-7 h-7" />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">AI Conversational Coach</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6">
                Engage in conversational shadowing with Ava, our custom AI tutor. Choose between formal, casual, or inquisitive conversation threads, record your response, and receive immediate fluency scoring mapped directly to the dialog.
              </p>
              <ul className="text-xs text-zinc-500 space-y-2 mb-8">
                <li className="flex items-center gap-2">✔ Contextual dialogue roleplay scenarios</li>
                <li className="flex items-center gap-2">✔ Automatic Text-to-Speech (TTS) reference playback</li>
                <li className="flex items-center gap-2">✔ Multi-model OpenRouter LLM fallback backend</li>
              </ul>
            </div>
            <div className="mt-auto" onClick={() => router.push("/ai-agent")}>
              <MagneticButton size="default" variant="secondary">
                Talk to Ava
              </MagneticButton>
            </div>
          </motion.div>

          {/* Card 4: Minimal Pairs & Spaced Repetition */}
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="group flex flex-col justify-between p-8 rounded-3xl border border-white/5 bg-zinc-900/40 backdrop-blur-sm hover:border-orange-500/30 transition-all duration-500 shadow-xl"
          >
            <div>
              <div className="w-14 h-14 rounded-2xl bg-orange-500/20 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-300">
                <Star className="text-orange-500 w-7 h-7" />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Practice Drills & Review</h3>
              <p className="text-zinc-400 text-sm leading-relaxed mb-6">
                Target common Indian English pronunciation confusion sets (like /v/ vs /w/ or retroflex stops) with minimal-pair drills. Track your weak sounds with a Spaced Repetition flashcard system using the SM-2 algorithm.
              </p>
              <ul className="text-xs text-zinc-500 space-y-2 mb-8">
                <li className="flex items-center gap-2">✔ AI-generated custom minimal-pair cards</li>
                <li className="flex items-center gap-2">✔ Smart interval review scheduling</li>
                <li className="flex items-center gap-2">✔ Target phone focus decks</li>
              </ul>
            </div>
            <div className="mt-auto" onClick={() => router.push("/drills")}>
              <MagneticButton size="default" variant="primary">
                Open Practice Drills
              </MagneticButton>
            </div>
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="relative z-10 py-28 bg-zinc-950/60 border-t border-white/5 scroll-mt-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="text-4xl md:text-5xl font-light text-white mb-4">How Scoring Works</h2>
            <p className="font-mono text-orange-500/80 tracking-wider uppercase text-sm">/ UNDER THE HOOD OF VOICESCORE</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
            <div className="flex flex-col gap-4">
              <div className="font-mono text-orange-500 text-4xl font-light">01</div>
              <h3 className="text-xl font-semibold text-white">Spectral Trimming & VAD</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                Your audio input is transcoded to 16kHz mono PCM WAV. An initial FFT spectral subtraction reduces background hum, and a Silero Voice Activity Detector (VAD) crops trailing and leading silence to isolate your speech frames.
              </p>
            </div>
            <div className="flex flex-col gap-4">
              <div className="font-mono text-orange-500 text-4xl font-light">02</div>
              <h3 className="text-xl font-semibold text-white">Contrastive Frame Alignment</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                A 96M-parameter Wav2Vec2 encoder extracts acoustic vectors for each 20ms frame. Both these audio frames and target phonemes are projected into a shared 256-dimensional metric space and aligned using CPU-based Forced Alignment.
              </p>
            </div>
            <div className="flex flex-col gap-4">
              <div className="font-mono text-orange-500 text-4xl font-light">03</div>
              <h3 className="text-xl font-semibold text-white">Goodness-of-Pronunciation</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                We calculate the log-likelihood ratio (LLR) for each aligned phoneme frame against all alternative target paths. Correct sounds match closely, while substitutions or omissions skew the score downward.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Architecture Section */}
      <section id="tech" className="relative z-10 py-28 border-t border-white/5 scroll-mt-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-light text-white mb-4">Technical Architecture</h2>
            <p className="font-mono text-orange-500/80 tracking-wider uppercase text-sm">/ SCIENTIFIC MODEL METRICS</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16 text-center">
            <div className="p-8 rounded-2xl bg-zinc-900/30 border border-white/5">
              <div className="text-4xl font-bold text-orange-500 mb-2">14.17%</div>
              <div className="text-xs font-mono text-zinc-500 uppercase tracking-widest">Validation PER</div>
            </div>
            <div className="p-8 rounded-2xl bg-zinc-900/30 border border-white/5">
              <div className="text-4xl font-bold text-orange-500 mb-2">133,737</div>
              <div className="text-xs font-mono text-zinc-500 uppercase tracking-widest">Extended Lexicon Words</div>
            </div>
            <div className="p-8 rounded-2xl bg-zinc-900/30 border border-white/5">
              <div className="text-4xl font-bold text-orange-500 mb-2">60</div>
              <div className="text-xs font-mono text-zinc-500 uppercase tracking-widest">Active IPA Tokens</div>
            </div>
          </div>

          <div className="p-8 rounded-3xl bg-zinc-900/20 border border-white/5 backdrop-blur-xs max-w-3xl mx-auto">
            <h3 className="text-lg font-semibold text-white mb-4 text-center">Model Training details</h3>
            <p className="text-sm text-zinc-400 leading-relaxed mb-6">
              Our acoustic model (checkpoint: <code>MihirRPatil/nptel-asr-phoneme-v3</code>) was trained on a balanced mixture of Indian English speech: 50% AI4Bharat NPTEL lectures, 20% Common Voice India, 10% Svarah conversational speech, 10% OpenSLR-104 MUCS regional accents, and 10% Eka Care medical queries.
            </p>
            <ul className="text-xs text-zinc-500 space-y-2">
              <li className="flex justify-between border-b border-white/5 pb-2">
                <span>Optimiser</span>
                <span>AdamW (Learning Rate: 2e-5)</span>
              </li>
              <li className="flex justify-between border-b border-white/5 pb-2">
                <span>Hardware</span>
                <span>NVIDIA RTX A6000 (48GB VRAM)</span>
              </li>
              <li className="flex justify-between pb-2">
                <span>Phoneme Set</span>
                <span>Dental/Retroflex stops, palatalized/labialized variants</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="relative z-10 py-28 px-6 text-center max-w-4xl mx-auto border-t border-white/5">
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
