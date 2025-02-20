import React from 'react';

type InputVideoProps = {
  setVideoUrl: (url: string) => void;
};

const InputVideo: React.FC<InputVideoProps> = ({ setVideoUrl }) => {
  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      const file = event.target.files[0];
      const fileUrl = URL.createObjectURL(file);
      setVideoUrl(fileUrl);
    }
  };

  return (
    <input
      type="file"
      accept="video/*"
      onChange={handleFileChange}
      className="mb-4 p-2 border rounded-md bg-gray-800 border-gray-700 text-white"
    />
  );
};

export default InputVideo;
