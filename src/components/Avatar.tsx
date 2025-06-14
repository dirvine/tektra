import { useEffect, useRef } from "react";
import { AvatarController } from "../avatar/AvatarController";
import "../styles/Avatar.css";

interface AvatarProps {
  currentViseme: string;
  isListening: boolean;
  isSpeaking: boolean;
}

const Avatar: React.FC<AvatarProps> = ({ currentViseme, isListening, isSpeaking }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const controllerRef = useRef<AvatarController | null>(null);

  useEffect(() => {
    if (canvasRef.current && !controllerRef.current) {
      controllerRef.current = new AvatarController(canvasRef.current);
      controllerRef.current.start();
    }

    return () => {
      if (controllerRef.current) {
        controllerRef.current.stop();
      }
    };
  }, []);

  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.setViseme(currentViseme);
    }
  }, [currentViseme]);

  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.setListening(isListening);
    }
  }, [isListening]);

  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.setSpeaking(isSpeaking);
    }
  }, [isSpeaking]);

  return (
    <div className="avatar-container">
      <canvas
        ref={canvasRef}
        width={400}
        height={400}
        className="avatar-canvas"
      />
      <div className="avatar-status">
        {isListening && <span className="status-indicator listening">Listening...</span>}
        {isSpeaking && <span className="status-indicator speaking">Speaking...</span>}
      </div>
    </div>
  );
};

export default Avatar;