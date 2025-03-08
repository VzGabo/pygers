import React, { useRef } from 'react';
import Webcam from 'react-webcam';

interface Props {}

const WebcamComponent: React.FunctionComponent<Props> = () => {
  const webcamRef = useRef<Webcam>(null);

  const handlePause = () => {
    if (webcamRef.current) {
      webcamRef.current.video?.pause();
    }
  };

  const handleResume = () => {
    if (webcamRef.current) {
      webcamRef.current.video?.play();
    }
  };

  const handleStop = () => {
    if (webcamRef.current) {
      const video = webcamRef.current.video;
      if (video) {
        video.srcObject = null;
      }
    }
  };

  return (
    <div>
      <h1>Visualización de la Cámara</h1>
      <Webcam
        audio={false}
        height={720}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={1280}
        videoConstraints={{
          width: 1280,
          height: 720,
          facingMode: 'user',
        }}
      />
      <div>
        <button onClick={handlePause}>Pausar</button>
        <button onClick={handleResume}>Reanudar</button>
        <button onClick={handleStop}>Detener</button>
      </div>
    </div>
  );
};

export default WebcamComponent;
