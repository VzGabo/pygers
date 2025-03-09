import React, { useRef } from 'react';
import Webcam from 'react-webcam';
import Navbar from './Navbar';

interface Props {}

const WebcamComponent: React.FunctionComponent<Props> = () => {
  const webcamRef = useRef<Webcam>(null);

  const handleStart = () => {
    if (webcamRef.current) {
      const video = webcamRef.current.video;
      if (video) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then((stream) => {
            video.srcObject = stream;
          })
          .catch((error) => {
            console.error('Error accessing webcam:', error);
          });
      }
    }
  };

  const handlePause = async () => {
    if (webcamRef.current) {
      webcamRef.current.video?.pause();
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const blob = await fetch(imageSrc).then(res => res.blob());
        const file = new File([blob], "screenshot.jpg", { type: "image/jpeg" });

        const formData = new FormData();
        formData.append("file", file);

        try {
          const response = await fetch('http://localhost:8000/compare-faces', {
            method: 'POST',
            body: formData,
          });
          const result = await response.json();
          console.log('Comparison result:', result);
        } catch (error) {
          console.error('Error sending image to backend:', error);
        }
      }
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
    <>
      <div>
        <Navbar/>
        <h1 className="text-4xl mb-8 flex flex-col items-center justify-center text-white font-bold mt-4">Visualización de la Cámara</h1>
        <div className="flex items-center justify-center">
        <Webcam
          audio={false}
          height={720}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={800}
          videoConstraints={{
            width: 1280,
            height: 720,
            facingMode: 'user',
          }}
          className="border-4 border-orange-500"
        />
      </div>
        <div className="mt-4 flex justify-center space-x-4 m-5">
          <button onClick={handleStart} className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90 ">Iniciar</button>
          <button onClick={handleStop} className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90 ">Detener</button>
          <button onClick={handlePause} className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90 ">Pausar</button>
          <button onClick={handleResume} className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90 ">Reanudar</button>
        </div>
      </div>
      
    </>
  );
};

export default WebcamComponent;
