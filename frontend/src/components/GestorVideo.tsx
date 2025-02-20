import React, { useState } from 'react';
import ReactPlayer from 'react-player';

export default function GestorVideo() {
  const [videoUrl, setVideoUrl] = useState<string>('');

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      const fileUrl = URL.createObjectURL(file);
      setVideoUrl(fileUrl);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white">
      <h1 className="text-2xl font-bold mb-4 text-center">Busqueda por video</h1>
      <p
        className="mb-4 px-4 py-2 bg-white-500 hover:bg-white-600 text-white rounded-md"
        >
          Subir o Cambiar Video
      </p>
      <input
        type="file"
        accept="video/*"
        onChange={handleFileChange}
        className="mb-4 p-2 border rounded-md bg-gray-800 border-gray-700 text-white"
    />
    <h2 className="text-xl font-semibold mb-2 text-center">Video Actual</h2>
    {videoUrl && (
        <div className="w-full max-w-md flex justify-center">
            <ReactPlayer url={videoUrl} controls width="100%" height="auto" className="rounded-lg shadow-lg" />
        </div>
      )}
    </div>
  );
}
