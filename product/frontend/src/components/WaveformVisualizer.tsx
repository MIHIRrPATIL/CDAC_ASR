"use client";

import React, { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  stream: MediaStream | null;
  isRecording: boolean;
}

export default function WaveformVisualizer({
  stream,
  isRecording,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isRecording || !stream || !canvasRef.current) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set up Audio Context and Analyser
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    const analyser = audioContext.createAnalyser();
    
    analyser.fftSize = 256;
    source.connect(analyser);
    
    audioCtxRef.current = audioContext;
    analyserRef.current = analyser;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;

      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      // Black backing with slight transparency for a neon trail effect
      ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
      ctx.fillRect(0, 0, width, height);

      const barWidth = (width / bufferLength) * 1.5;
      let barHeight;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        barHeight = dataArray[i] * 0.45; // Scale height

        // Premium styling: Gradient from orange to violet
        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        gradient.addColorStop(0, "#f97316"); // Orange
        gradient.addColorStop(0.5, "#a855f7"); // Purple
        gradient.addColorStop(1, "#ec4899"); // Pink

        ctx.fillStyle = gradient;

        // Draw symmetrical rounded bars starting from the center height
        const yPos = (height - barHeight) / 2;
        ctx.beginPath();
        if (ctx.roundRect) {
          ctx.roundRect(x, yPos, barWidth - 2, barHeight, 4);
        } else {
          ctx.rect(x, yPos, barWidth - 2, barHeight);
        }
        ctx.fill();

        x += barWidth;
      }
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
    };
  }, [isRecording, stream]);

  return (
    <div className="relative w-full h-16 bg-black rounded-xl overflow-hidden border border-border/20 shadow-inner flex items-center justify-center">
      <canvas
        ref={canvasRef}
        width={400}
        height={64}
        className="w-full h-full block"
      />
      {!isRecording && (
        <span className="absolute text-xs font-mono tracking-wider text-muted-foreground uppercase opacity-60">
          MICROPHONE READY
        </span>
      )}
    </div>
  );
}
