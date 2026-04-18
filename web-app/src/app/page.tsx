"use client";

import { useRef, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, useScroll, useSpring, useTransform } from "framer-motion";
import { CustomCursor } from "@/components/custom-cursor";
import { GrainOverlay } from "@/components/grain-overlay";
import { MagneticButton } from "@/components/magnetic-button";
import EtherBackground from "@/components/EtherBackground";
import { Activity, Mic, BarChart3, Rocket, Target, Zap } from "lucide-react";

import { useAuth } from "@/lib/auth-context";

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [currentSection, setCurrentSection] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);
    if (isAuthenticated) {
      router.replace("/dashboard");
    }
  }, [isAuthenticated, router]);

  useEffect(() => {
    const handleWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        if (!scrollContainerRef.current) return;
        
        // Prevent default vertical scroll only if we are horizontally scrolling
        e.preventDefault();
        
        scrollContainerRef.current.scrollBy({
          left: e.deltaY,
          behavior: "auto",
        });

        const sectionWidth = scrollContainerRef.current.offsetWidth;
        const newSection = Math.round(scrollContainerRef.current.scrollLeft / sectionWidth);
        if (newSection !== currentSection) {
          setCurrentSection(newSection);
        }
      }
    };

    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
    }

    return () => {
      if (container) {
        container.removeEventListener("wheel", handleWheel);
      }
    };
  }, [currentSection]);

  const sections = [
    {
      id: "hero",
      content: (
        <div className="max-w-4xl">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-6 inline-block rounded-full border border-orange-500/30 bg-orange-500/10 px-4 py-1.5 backdrop-blur-md"
          >
            <p className="font-mono text-xs text-orange-400 uppercase tracking-widest">
              AI-Powered Pronunciation Scoring
            </p>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="mb-8 font-sans text-7xl font-light leading-[1.05] tracking-tight text-white md:text-8xl lg:text-9xl"
          >
            Master your <br />
            <span className="text-orange-500">spoken fluency.</span>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="mb-10 max-w-xl text-lg leading-relaxed text-zinc-400 md:text-xl"
          >
            Analyze your pronunciation with clinical precision. Get real-time feedback 
            on phonemes, pitch, and duration using our advanced neural pipeline.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="flex flex-col gap-4 sm:flex-row sm:items-center"
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
        </div>
      )
    },
    {
      id: "features",
      content: (
        <div className="w-full max-w-6xl">
          <div className="mb-16">
            <h2 className="text-5xl font-light text-white mb-4">Precision Metrics</h2>
            <p className="font-mono text-orange-500/60 tracking-wider">/ SCIENTIFIC APPROACH TO FLUENCY</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {[
              { icon: Target, title: "Phonetic Accuracy", desc: "Detailed scoring for every phoneme in your speech." },
              { icon: Zap, title: "Pitch Analysis", desc: "Visualize your intonation curves against native models." },
              { icon: Activity, title: "Duration & Rhythm", desc: "Master the timing and stress patterns of natural speech." }
            ].map((f, i) => (
              <motion.div 
                key={i}
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.2 }}
                className="group p-8 rounded-3xl border border-white/5 bg-white/5 backdrop-blur-sm hover:border-orange-500/30 transition-all duration-500"
              >
                <div className="w-12 h-12 rounded-2xl bg-orange-500/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <f.icon className="text-orange-500" size={24} />
                </div>
                <h3 className="text-2xl font-medium text-white mb-4">{f.title}</h3>
                <p className="text-zinc-500 leading-relaxed">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: "cta",
      content: (
        <div className="text-center max-w-3xl mx-auto">
          <h2 className="text-6xl font-light text-white mb-8">Ready to transform your speech?</h2>
          <p className="text-xl text-zinc-400 mb-12">Join thousands of students mastering their spoken English with LinguaScore.</p>
          <div className="flex justify-center" onClick={() => router.push("/auth")}>
             <MagneticButton size="lg" variant="primary">
                Create Free Account
              </MagneticButton>
          </div>
        </div>
      )
    }
  ];

  return (
    <main className="relative h-screen w-full overflow-hidden bg-black selection:bg-orange-500/30">
      <CustomCursor />
      <GrainOverlay />
      <EtherBackground />

      <div
        ref={scrollContainerRef}
        className={`relative z-10 flex h-screen overflow-x-auto overflow-y-hidden transition-opacity duration-1000 ${isLoaded ? "opacity-100" : "opacity-0"}`}
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {sections.map((section, index) => (
          <section
            key={section.id}
            className="flex min-h-screen w-screen shrink-0 items-center px-6 md:px-12 lg:px-24"
          >
            {section.content}
          </section>
        ))}
      </div>

      {/* Navigation Progress Dots */}
      <div className="fixed bottom-12 left-1/2 -translate-x-1/2 z-20 flex gap-4">
        {sections.map((_, i) => (
          <div 
            key={i}
            className={`w-2 h-2 rounded-full transition-all duration-500 ${currentSection === i ? "bg-orange-500 w-8" : "bg-white/20 hover:bg-white/40"}`}
          />
        ))}
      </div>

      <style jsx global>{`
        div::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </main>
  );
}
