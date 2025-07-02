import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mic, MicOff, Volume2, Loader2, Zap } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

interface ConversationStatusProps {
  className?: string;
}

type ConversationMode = 
  | 'Idle'
  | 'WakeWordDetected' 
  | 'ActiveListening'
  | 'Processing'
  | 'Responding'
  | 'WaitingForUser';

const ConversationStatus: React.FC<ConversationStatusProps> = ({ className = '' }) => {
  const [mode, setMode] = useState<ConversationMode>('Idle');
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [aiText, setAiText] = useState('');
  const [showWakeWordPulse, setShowWakeWordPulse] = useState(false);

  useEffect(() => {
    // Set up event listeners
    const unlistenWakeWord = listen('conversation-wake-word-detected', (event) => {
      console.log('Wake word detected!', event);
      setMode('WakeWordDetected');
      setShowWakeWordPulse(true);
      setTimeout(() => setShowWakeWordPulse(false), 2000);
    });

    const unlistenUserInput = listen('conversation-user-input', (event: any) => {
      console.log('User input:', event);
      setTranscribedText(event.payload.text);
      setMode('Processing');
    });

    const unlistenAiResponding = listen('conversation-ai-responding', (event: any) => {
      console.log('AI responding:', event);
      setAiText(event.payload.text);
      setMode('Responding');
      setIsSpeaking(true);
    });

    const unlistenConversationEnd = listen('conversation-ended', () => {
      console.log('Conversation ended');
      setMode('Idle');
      setTranscribedText('');
      setAiText('');
    });

    const unlistenTtsSpeaking = listen('tts-speaking-started', () => {
      setIsSpeaking(true);
    });

    const unlistenTtsFinished = listen('tts-speaking-finished', () => {
      setIsSpeaking(false);
      if (mode === 'Responding') {
        setMode('WaitingForUser');
      }
    });

    // Start always-listening mode
    invoke('start_always_listening')
      .then(() => {
        console.log('Always-listening mode started');
        setIsListening(true);
      })
      .catch((error) => {
        console.error('Failed to start always-listening:', error);
      });

    // Cleanup
    return () => {
      Promise.all([
        unlistenWakeWord,
        unlistenUserInput,
        unlistenAiResponding,
        unlistenConversationEnd,
        unlistenTtsSpeaking,
        unlistenTtsFinished,
      ]).then((unlisteners) => {
        unlisteners.forEach((unlisten) => unlisten());
      });
    };
  }, []);

  const getModeDisplay = () => {
    switch (mode) {
      case 'Idle':
        return {
          icon: <Mic className="w-5 h-5" />,
          text: 'Say "Tektra" to start',
          color: 'text-text-secondary',
          bgColor: 'bg-surface',
          animate: false,
        };
      case 'WakeWordDetected':
        return {
          icon: <Zap className="w-5 h-5" />,
          text: 'Tektra is listening...',
          color: 'text-accent',
          bgColor: 'bg-accent/20',
          animate: true,
        };
      case 'ActiveListening':
        return {
          icon: <Mic className="w-5 h-5" />,
          text: 'Listening...',
          color: 'text-success',
          bgColor: 'bg-success/20',
          animate: true,
        };
      case 'Processing':
        return {
          icon: <Loader2 className="w-5 h-5 animate-spin" />,
          text: 'Thinking...',
          color: 'text-accent',
          bgColor: 'bg-accent/10',
          animate: false,
        };
      case 'Responding':
        return {
          icon: <Volume2 className="w-5 h-5" />,
          text: 'Speaking...',
          color: 'text-accent',
          bgColor: 'bg-accent/20',
          animate: true,
        };
      case 'WaitingForUser':
        return {
          icon: <Mic className="w-5 h-5" />,
          text: 'Your turn...',
          color: 'text-success',
          bgColor: 'bg-success/10',
          animate: false,
        };
    }
  };

  const modeDisplay = getModeDisplay();

  return (
    <div className={`${className}`}>
      {/* Status Bar */}
      <motion.div
        className={`
          flex items-center justify-between px-4 py-2 rounded-lg
          ${modeDisplay.bgColor} border border-border-primary
          transition-all duration-300
        `}
        animate={{
          scale: modeDisplay.animate ? [1, 1.02, 1] : 1,
        }}
        transition={{
          duration: 2,
          repeat: modeDisplay.animate ? Infinity : 0,
          ease: "easeInOut",
        }}
      >
        <div className="flex items-center space-x-3">
          <div className={`${modeDisplay.color}`}>
            {modeDisplay.icon}
          </div>
          <span className={`text-sm font-medium ${modeDisplay.color}`}>
            {modeDisplay.text}
          </span>
        </div>

        {/* Listening Indicator */}
        <div className="flex items-center space-x-2">
          {isListening && (
            <motion.div
              className="flex items-center space-x-1"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              {[...Array(3)].map((_, i) => (
                <motion.div
                  key={i}
                  className="w-1 h-3 bg-accent rounded-full"
                  animate={{
                    height: ['12px', '20px', '12px'],
                  }}
                  transition={{
                    duration: 1,
                    repeat: Infinity,
                    delay: i * 0.2,
                  }}
                />
              ))}
            </motion.div>
          )}

          {/* Speaking Indicator */}
          {isSpeaking && (
            <motion.div
              className="w-2 h-2 bg-accent rounded-full"
              animate={{
                scale: [1, 1.5, 1],
                opacity: [1, 0.5, 1],
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
              }}
            />
          )}
        </div>
      </motion.div>

      {/* Wake Word Pulse Effect */}
      <AnimatePresence>
        {showWakeWordPulse && (
          <motion.div
            className="absolute inset-0 pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2"
              initial={{ scale: 0, opacity: 1 }}
              animate={{ scale: 4, opacity: 0 }}
              transition={{ duration: 1.5, ease: "easeOut" }}
            >
              <div className="w-20 h-20 bg-accent rounded-full" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transcription Display */}
      <AnimatePresence>
        {(transcribedText || aiText) && (
          <motion.div
            className="mt-3 p-3 bg-surface/50 rounded-lg"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            {transcribedText && (
              <div className="mb-2">
                <span className="text-xs text-text-tertiary">You said:</span>
                <p className="text-sm text-text-primary">{transcribedText}</p>
              </div>
            )}
            {aiText && (
              <div>
                <span className="text-xs text-text-tertiary">Tektra:</span>
                <p className="text-sm text-text-primary line-clamp-3">{aiText}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Instructions */}
      {mode === 'Idle' && (
        <motion.div
          className="mt-3 text-xs text-text-tertiary text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <p>Tektra is always listening for the wake word.</p>
          <p>Just say "Tektra" followed by your question!</p>
        </motion.div>
      )}
    </div>
  );
};

export default ConversationStatus;