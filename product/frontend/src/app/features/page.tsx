"use client";

import { motion } from "framer-motion";
import { Rocket, Sparkles, Zap, Shield, Globe, Cpu } from "lucide-react";
import { CustomCursor } from "@/components/custom-cursor";
import { GrainOverlay } from "@/components/grain-overlay";
import EtherBackground from "@/components/EtherBackground";
import { MagneticButton } from "@/components/magnetic-button";
import { useRouter } from "next/navigation";

export default function FeaturesPage() {
  const router = useRouter();

  const features = [
    { icon: Zap, title: "Real-time Feedback", desc: "Instant corrections and visual heatmaps for your pronunciation." },
    { icon: Globe, title: "Multilingual Support", desc: "Master accents from across the globe with diverse native models." },
    { icon: Shield, title: "Clinical Precision", desc: "Phoneme-level analysis using state-of-the-art neural networks." },
    { icon: Cpu, title: "AI Learning Paths", desc: "GenAI driven curriculums tailored to your specific weaknesses." },
  ];

  return (
    <main className="relative min-h-screen w-full overflow-hidden bg-black text-white selection:bg-orange-500/30">
      <CustomCursor />
      <GrainOverlay />
      <EtherBackground />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 py-24 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="mb-8 p-4 rounded-3xl bg-orange-500/10 border border-orange-500/20 backdrop-blur-xl"
        >
          <Rocket className="text-orange-500 w-12 h-12" />
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-6xl md:text-8xl font-light tracking-tight mb-6"
        >
          Something <span className="text-orange-500 italic">extraordinary</span> <br />
          is coming soon.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.4 }}
          className="text-zinc-500 max-w-2xl text-lg md:text-xl mb-12 leading-relaxed"
        >
          We are refining our advanced AI features to give you the most accurate 
          speech analysis experience ever created. Stay tuned for the future of fluency.
        </motion.p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl mb-16">
          {features.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              className="p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-md text-left group hover:border-orange-500/30 transition-all duration-500"
            >
              <f.icon className="text-orange-500/60 mb-4 group-hover:text-orange-500 transition-colors" size={24} />
              <h3 className="text-lg font-medium mb-2">{f.title}</h3>
              <p className="text-sm text-zinc-500">{f.desc}</p>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          onClick={() => router.push("/dashboard")}
        >
          <MagneticButton size="lg" variant="primary">
            Back to Dashboard
          </MagneticButton>
        </motion.div>
      </div>
    </main>
  );
}
