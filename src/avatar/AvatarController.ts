export interface PhonemeData {
  phoneme: string;
  duration: number;
  timestamp: number;
}

interface AnimationFrame {
  viseme: string;
  intensity: number;
}

export class AvatarController {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private currentViseme: string = "neutral";
  private targetViseme: string = "neutral";
  private visemeBlend: number = 0;
  private animationQueue: AnimationFrame[] = [];
  private isListening: boolean = false;
  private isSpeaking: boolean = false;
  private animationId: number | null = null;
  
  // Avatar properties
  private centerX: number;
  private centerY: number;
  private headRadius: number = 120;
  private eyeSpacing: number = 40;
  private eyeY: number = -20;
  
  // Animation states
  private blinkTimer: number = 0;
  private isBlinking: boolean = false;
  private breathingPhase: number = 0;
  private listeningPulse: number = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.centerX = canvas.width / 2;
    this.centerY = canvas.height / 2;
  }

  start() {
    this.animate();
  }

  stop() {
    if (this.animationId !== null) {
      cancelAnimationFrame(this.animationId);
    }
  }

  setViseme(viseme: string) {
    this.targetViseme = viseme;
  }

  setListening(listening: boolean) {
    this.isListening = listening;
  }

  setSpeaking(speaking: boolean) {
    this.isSpeaking = speaking;
  }

  async playPhonemes(phonemes: PhonemeData[]) {
    const visemes = phonemes.map(p => ({
      viseme: this.phonemeToViseme(p.phoneme),
      duration: p.duration,
      timestamp: p.timestamp
    }));
    
    this.animationQueue = this.generateAnimationFrames(visemes);
  }

  private animate = () => {
    this.update();
    this.render();
    this.animationId = requestAnimationFrame(this.animate);
  };

  private update() {
    // Update breathing animation
    this.breathingPhase += 0.02;
    
    // Update listening pulse
    if (this.isListening) {
      this.listeningPulse += 0.1;
    }
    
    // Update blinking
    this.blinkTimer += 0.016;
    if (this.blinkTimer > 3 + Math.random() * 2) {
      this.isBlinking = true;
      this.blinkTimer = 0;
    }
    if (this.isBlinking && this.blinkTimer > 0.15) {
      this.isBlinking = false;
    }
    
    // Blend visemes
    if (this.currentViseme !== this.targetViseme) {
      this.visemeBlend += 0.1;
      if (this.visemeBlend >= 1) {
        this.currentViseme = this.targetViseme;
        this.visemeBlend = 0;
      }
    }
    
    // Process animation queue
    if (this.animationQueue.length > 0) {
      const frame = this.animationQueue.shift()!;
      this.setViseme(frame.viseme);
    }
  }

  private render() {
    const ctx = this.ctx;
    
    // Clear canvas
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    
    // Apply breathing effect
    const breathScale = 1 + Math.sin(this.breathingPhase) * 0.02;
    
    ctx.save();
    ctx.translate(this.centerX, this.centerY);
    ctx.scale(breathScale, breathScale);
    
    // Draw head outline
    ctx.strokeStyle = "#4a9eff";
    ctx.lineWidth = 3;
    
    if (this.isListening) {
      // Pulsing effect when listening
      const pulseSize = Math.sin(this.listeningPulse) * 10;
      ctx.shadowBlur = 20 + pulseSize;
      ctx.shadowColor = "#4a9eff";
    }
    
    ctx.beginPath();
    ctx.arc(0, 0, this.headRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw eyes
    this.drawEyes(ctx);
    
    // Draw mouth based on current viseme
    this.drawMouth(ctx);
    
    ctx.restore();
    
    // Draw status indicators
    if (this.isSpeaking) {
      this.drawSpeakingIndicator(ctx);
    }
  }

  private drawEyes(ctx: CanvasRenderingContext2D) {
    const eyeHeight = this.isBlinking ? 2 : 15;
    
    // Left eye
    ctx.fillStyle = "#4a9eff";
    ctx.fillRect(-this.eyeSpacing - 15, this.eyeY - eyeHeight/2, 30, eyeHeight);
    
    // Right eye
    ctx.fillRect(this.eyeSpacing - 15, this.eyeY - eyeHeight/2, 30, eyeHeight);
  }

  private drawMouth(ctx: CanvasRenderingContext2D) {
    ctx.strokeStyle = "#4a9eff";
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    
    const mouthY = 40;
    const mouthWidth = 60;
    
    switch (this.currentViseme) {
      case "neutral":
        // Straight line
        ctx.beginPath();
        ctx.moveTo(-mouthWidth/2, mouthY);
        ctx.lineTo(mouthWidth/2, mouthY);
        ctx.stroke();
        break;
        
      case "A":
      case "aa":
        // Open wide
        ctx.beginPath();
        ctx.ellipse(0, mouthY, mouthWidth/2, 20, 0, 0, Math.PI * 2);
        ctx.stroke();
        break;
        
      case "O":
      case "oh":
        // Round open
        ctx.beginPath();
        ctx.arc(0, mouthY, 25, 0, Math.PI * 2);
        ctx.stroke();
        break;
        
      case "E":
      case "ee":
        // Wide smile
        ctx.beginPath();
        ctx.moveTo(-mouthWidth/2, mouthY);
        ctx.bezierCurveTo(-mouthWidth/3, mouthY - 10, mouthWidth/3, mouthY - 10, mouthWidth/2, mouthY);
        ctx.stroke();
        break;
        
      case "U":
      case "oo":
        // Small round
        ctx.beginPath();
        ctx.arc(0, mouthY, 15, 0, Math.PI * 2);
        ctx.stroke();
        break;
        
      case "M":
      case "B":
      case "P":
        // Closed lips
        ctx.beginPath();
        ctx.moveTo(-mouthWidth/3, mouthY);
        ctx.lineTo(mouthWidth/3, mouthY);
        ctx.stroke();
        break;
        
      default:
        // Default neutral
        ctx.beginPath();
        ctx.moveTo(-mouthWidth/2, mouthY);
        ctx.lineTo(mouthWidth/2, mouthY);
        ctx.stroke();
    }
  }

  private drawSpeakingIndicator(ctx: CanvasRenderingContext2D) {
    ctx.save();
    ctx.translate(this.centerX, this.centerY + this.headRadius + 30);
    
    // Draw sound waves
    ctx.strokeStyle = "#4a9eff";
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6;
    
    for (let i = 0; i < 3; i++) {
      const offset = i * 15;
      const phase = this.breathingPhase + i * 0.5;
      const amplitude = Math.sin(phase) * 5;
      
      ctx.beginPath();
      ctx.moveTo(-30 + offset, amplitude);
      ctx.quadraticCurveTo(-15 + offset, -amplitude, 0 + offset, amplitude);
      ctx.quadraticCurveTo(15 + offset, -amplitude, 30 + offset, amplitude);
      ctx.stroke();
    }
    
    ctx.restore();
  }

  private phonemeToViseme(phoneme: string): string {
    // Simplified phoneme to viseme mapping
    const mapping: { [key: string]: string } = {
      "AA": "A", "AE": "A", "AH": "A", "AO": "O", "AW": "O",
      "AY": "A", "B": "B", "CH": "ch", "D": "D", "DH": "th",
      "EH": "E", "ER": "E", "EY": "E", "F": "F", "G": "G",
      "HH": "H", "IH": "I", "IY": "ee", "JH": "ch", "K": "K",
      "L": "L", "M": "M", "N": "N", "NG": "N", "OW": "oh",
      "OY": "oh", "P": "P", "R": "R", "S": "S", "SH": "ch",
      "T": "D", "TH": "th", "UH": "U", "UW": "oo", "V": "F",
      "W": "oo", "Y": "ee", "Z": "S", "ZH": "ch"
    };
    
    return mapping[phoneme.toUpperCase()] || "neutral";
  }

  private generateAnimationFrames(visemes: any[]): AnimationFrame[] {
    const frames: AnimationFrame[] = [];
    const frameRate = 60;
    
    for (const viseme of visemes) {
      const frameCount = Math.floor(viseme.duration * frameRate);
      for (let i = 0; i < frameCount; i++) {
        frames.push({
          viseme: viseme.viseme,
          intensity: 1.0
        });
      }
    }
    
    return frames;
  }
}