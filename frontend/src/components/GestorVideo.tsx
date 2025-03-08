import { useState } from 'react';
import ReactPlayer from 'react-player';
import InputVideo from './ButtonVideo';

export default function GestorVideo() {
  const [videoUrl, setVideoUrl] = useState<string>('');

  return (
    <div className="flex flex-col items-center justify-center min-h-screen text-white">
      <h1 className="text-2xl font-bold mb-6 text-center">Busqueda por video</h1>
      
      
      <InputVideo setVideoUrl={setVideoUrl} />
      
      {videoUrl && (
        <div className="w-full max-w-md flex justify-center">
          <ReactPlayer url={videoUrl} controls width="100%" height="auto" className="rounded-lg shadow-lg" />
        </div>
      )}
    </div>
  );
}
