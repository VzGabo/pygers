import SearchIcon from "../assets/icons/SearchIcon";
import { SetFile } from "../types/State";
import Button from "./Button";

interface Props {
  files: File[],
  setFiles: SetFile,
}

export default function Images({ files, setFiles }: Props) {
  const handleRemoveImage = (indexToRemove: number) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };
  return (
    <>
      <div id="images_list" className="relative flex justify-center items-center m-10 mb-3">
        {
          files.map((file, index) => (
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
          ))
        }
      </div>
      <div 
        id="buttons_container" 
        className="w-1/3 flex m-auto gap-3"
      >
        <Button
          type="outline"
          iconColor="primary"
          element="file"
          background="primary"
          setFiles={setFiles}
        >
          Sube mas sospechosos
        </Button>
        <Button
          element="button"
          background="primary"
          setFiles={setFiles}
        >
          <SearchIcon color="white" />
          <p>Buscar sospechosos</p>
        </Button>
      </div>
    </>
  )
}
