import Navbar from './components/Navbar'
import Button from './components/Button'
import IconImage from './components/upload'
import { useState } from 'react'
// import  from './components/orden'

export default function App() {
  // crear usestate
  const [files, setFiles] = useState<File[]>([])
  const handleRemoveImage = (indexToRemove: number) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };
  // psar el ste file al boton como parametro
  // y qeu el bonotn le pase el set file a guardar
  return (
    <main>
      <Navbar />
      <main className='flex justify-center mt-10'>
        {
          files.length === 0
          ?
          <>
            <Button background='primary' setFiles={setFiles} />
            <IconImage />
          </>
          :
          <>
            <div className="relative flex justify-center items-center">
            {files.map((file, index) => (
              <div key={index} className="relative">
                <img
                  className='w-34 m-2 rounded'
                  src={URL.createObjectURL(file)}
                  alt={`Imagen subida ${index + 1}`}
                />
                <button
                  className="absolute top-0 right-0 bg-red-500 text-white p-1 rounded-full"
                  onClick={() => handleRemoveImage(index)}
                >
                  X
                </button>
              </div>
            ))}
          </div>
          </>
          //mostrar imagen y guardar iamgen en una carpeta llamada assets/img
        }
      </main>
    </main>
  )
}

