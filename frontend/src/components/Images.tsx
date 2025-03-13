// import { SetFile } from "../types/State";
// import Button from "./Button";
// import SearchIcon from "../assets/icons/SearchIcon";
// import { Link } from 'react-router-dom';


// interface Props {
//   files: File[],
//   setFiles: SetFile,
// }

// export default function Image({ files, setFiles }: Props) {
//   const handleRemoveImage = (indexToRemove: number) => {
//     setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
//   };
//   return (
//     <>
//       <div id="file" className="relative flex justify-center items-center m-10 mb-3">
//         {
//           files.map((file, index) => (
//             <div key={index} className="relative">
//               <img
//                 className='w-34 m-2 rounded'
//                 src={URL.createObjectURL(file)}
//                 alt={`Imagen subida ${index + 1}`}
//               />
//               <button
//                 className="absolute top-0 right-0 bg-red-500 text-white p-2 rounded-full w-10 cursor-pointer transition hover:bg-red-400"
//                 onClick={() => handleRemoveImage(index)}
//               >
//                 X
//               </button>
//             </div>
//           ))
//         }
//       </div>
//       <div 
//         id="buttons_container" 
//         className="w-1/3 flex m-auto gap-3"
//       >
//         <Button
//           type="outline"
//           iconColor="primary"
//           element="file"
//           background="primary"
//           setFiles={setFiles}
//         >
//           Sube mas sospechosos
//         </Button>
//         <Button
//           element="button"
//           background="primary"
//           setFiles={setFiles}
//         >
//           <SearchIcon color="white" />
//           <Link to="/Opciones">Buscar sospechosos</Link>
//         </Button>
//       </div>
//     </>
//   )
// }
import React from 'react';
import Button from './Button';
import SearchIcon from '../assets/icons/SearchIcon';
import { useNavigate } from 'react-router-dom';
import { SetFile } from "../types/State";

interface Props {
  files: File[],
  setFiles: SetFile,
  type?: string;
  iconColor?: string;
  element: 'button' | 'file';
  background?: string;
  onChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onClick?: () => void | Promise<void>; // Agregar onClick como opcional
  children: React.ReactNode;
}

const Images: React.FC<Props> = ({ files, setFiles }) => {

  const navigate = useNavigate()

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newFiles = Array.from(event.target.files || []);
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
  };

  const handleRemoveImage = (indexToRemove: number) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };

  const handleUploadImages = async () => {
    console.log(files);
    if (files.length > 0) {
      const formData = new FormData();
      files.forEach(file => {
        formData.append("files", file);
      });

      try {
        const response = await fetch('http://localhost:8000/upload-faces', {
          method: 'POST',
          body: formData,
        });
        const result = await response.json();
        console.log('Upload result:', result);
        navigate("/Opciones")
      } catch (error) {
        console.error('Error uploading files to backend:', error);
      }
    } else {
      console.log('No files to upload');
    }
  };

  return (
    <>
      <div id="file" className="relative flex justify-center items-center m-10 mb-3 flex-wrap">
        {files.map((file, index) => (
          <div key={index} className="relative">
            <img
              className='w-34 m-2 rounded'
              src={URL.createObjectURL(file)}
              alt={`Imagen subida ${index + 1}`}
            />
            <button
              className="absolute top-0 right-0 bg-red-500 text-white p-2 rounded-full w-10 cursor-pointer transition hover:bg-red-400"
              onClick={() => handleRemoveImage(index)}
            >
              X
            </button>
          </div>
        ))}
      </div>
      <div
        id="buttons_container"
        className="w-1/3 flex m-auto gap-3"
      >
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="file-input"
        />
        <label htmlFor="file-input">
          <Button
            type="outline"
            iconColor="primary"
            element="file"
            background="primary"
            setFiles={setFiles}
          >
            Sube más sospechosos
          </Button>
        </label>
        <Button
          element="button"
          background="primary"
          setFiles={setFiles}
          onClick={handleUploadImages}
        >
          <SearchIcon color="white" />
          Buscar sospechosos
        </Button>
      </div>
    </>
  );
};

export default Images;