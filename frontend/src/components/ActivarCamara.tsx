import { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import Navbar from './Navbar';

export default function WebcamComponent() {
  const webcamRef = useRef<Webcam>(null);
  interface ComparisonResult {
    message: string;
    results?: { known_face_path: string; distance: number }[];
  }

  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);
  const [imagePaths, setImagePaths] = useState<string[]>([]);

  useEffect(() => {
    const interval = setInterval(captureFrame, 1000); // Cambia el intervalo según tus necesidades (en milisegundos)
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    fetchImages();
  }, []);

  const fetchImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/upload-faces');
      const data = await response.json();
      setImagePaths(data.path_images);
    } catch (error) {
      console.error('Error fetching images:', error);
    }
  };

  const captureFrame = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        const blob = await fetch(imageSrc).then(res => res.blob());
        const file = new File([blob], "frame.jpg", { type: "image/jpeg" });

        const formData = new FormData();
        formData.append("file", file);

        try {
          const response = await fetch('http://localhost:8000/compare-faces', {
            method: 'POST',
            body: formData,
          });
          const result = await response.json();
          setComparisonResult(result);
          if(result?.results) {
            handleStop()
          }
          console.log('Frame sent successfully:', result);
        } catch (error) {
          console.error('Error sending frame to backend:', error);
        }
      }
    }
  };

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
    <>
      <div>
        <Navbar />
        <h1 className="text-4xl m-10 justify-center text-white font-bold">
          Visualización de la Cámara
        </h1>
        <div className="flex flex-row justify-center space-x-8">
          <div className="flex flex-col items-center">
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
              onUserMedia={() => console.log('User media loaded')}
            />
            <div className="mt-4 flex justify-center space-x-4 m-5">
              <button
                onClick={handleStart}
                className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90"
              >
                Iniciar
              </button>
              <button
                onClick={handleStop}
                className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90"
              >
                Detener
              </button>
              <button
                onClick={handlePause}
                className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90"
              >
                Pausar
              </button>
              <button
                onClick={handleResume}
                className="w-48 bg-orange-500 hover:bg-white text-white hover:text-orange-500 font-bold py-2 px-4 rounded text-center cursor-pointer transition duration-200 active:scale-90"
              >
                Reanudar
              </button>
            </div>
          </div>
          {/* Mostrar resultados de comparación */}
          {comparisonResult && (
            <div className="flex flex-col items-center">
              <p className="text-white font-bold text-center text-2xl w-[60%]">
                {comparisonResult.message}
              </p>
              {comparisonResult.results && comparisonResult.results.length > 0 && (
                <div className="flex flex-col items-center">
                  <img
                    src={`http://localhost:8000/${comparisonResult.results[0].known_face_path.replace(/\\/g, '/')}`}
                    alt="Known face"
                    className="mt-4 border-4 border-orange-500"
                    width={200}
                    height={200}
                  />
                  <p className="text-white mt-2 text-center">
                    Porcentaje de aproximación: {((1 - comparisonResult.results[0].distance) * 100).toFixed(2)}%
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
